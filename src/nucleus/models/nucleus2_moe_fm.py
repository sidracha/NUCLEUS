from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
from torch.profiler import record_function

from nucleus.data.batching import CollatedBatch
from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.fm_time_embed import FMTimeEmbed
from nucleus.layers.patching import LinearEmbed
from nucleus.models._api import register_model
from nucleus.models.nucleus2_moe import Nucleus2MoE, Nucleus2MoEConfig, _config_from_dict, get_dtype


@register_model("nucleus2_moe_fm")
class Nucleus2MoEFlowMatching(Nucleus2MoE):
    """
    Nucleus2MoE with a dedicated per-block FiLM layer for flow time t.
    Context and noisy frames are stacked along time and each frame is its own token.
    """

    def __init__(self, config: Nucleus2MoEConfig):
        super().__init__(config)
        # a scalar t can't feed AdaptiveLayerNorm directly: its input LayerNorm over a
        # single feature always outputs 0, which silently erased t
        self.time_embed = FMTimeEmbed(config.embed_dim)
        self.t_norms = nn.ModuleList([
            AdaptiveLayerNorm(config.embed_dim, config.embed_dim, dtype=get_dtype(config.activation_dtype))
            for _ in range(config.processor_blocks)
        ])

    def forward(self, batch: CollatedBatch):
        raise NotImplementedError("Call step(input, sim_params, t) directly.")

    def _network_frames(self, input: torch.Tensor, num_context_frames: int,
                        condition: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        # returns the frames fed to the network and how many leading frames to drop before decoding
        if condition is not None:
            input = torch.cat([input, condition[:, None].expand(-1, input.shape[1], -1, -1, -1)], dim=-1)
        return input, num_context_frames

    def step(self, input: torch.Tensor, sim_params: torch.Tensor, t: torch.Tensor, T_in: int = 0,
             condition: Optional[torch.Tensor] = None):
        """
        input:     (B, T_in + T_out, H, W, C)  context frames followed by the noisy frames zt
        sim_params:(B, 13)
        t:         (B, 1)  flow time in [0, 1]
        T_in:      number of context frames — output is (B, T_out, H, W, C)
        condition: optional static maps (B, H, W, K), injected as config.condition_injection says
        """
        assert input.dtype == torch.float32
        assert sim_params.dtype == torch.float32
        assert t.dtype == torch.float32

        concat_condition = condition if self.concatenates_condition else None
        frames, num_dropped_frames = self._network_frames(input, T_in, concat_condition)
        _, _, h, w, _ = frames.shape

        with record_function("encode"):
            x = embed = self._encode(frames, condition)

        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(
                    embed_t, embed_h, embed_w
                )[None, :, :, :, None, :]

        time_features = self.time_embed(t)

        moe_outputs = []
        for idx, (blk, t_norm) in enumerate(zip(self.blocks, self.t_norms)):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                x = t_norm(x, time_features)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
            x = self.debed(x[:, num_dropped_frames:, ...], target_shape=(h, w))

        return x.to(torch.float32), moe_outputs


@dataclass
class Nucleus2MoEFMConcatConfig(Nucleus2MoEConfig):
    num_context_frames: int = 4


def _concat_config_from_dict(config_dict: dict) -> Nucleus2MoEFMConcatConfig:
    return _config_from_dict(config_dict, Nucleus2MoEFMConcatConfig)


@register_model("nucleus2_moe_fm_concat")
class Nucleus2MoEFlowMatchingConcat(Nucleus2MoEFlowMatching):
    """
    Flow matching with channel-concatenated conditioning: each noisy frame is paired with
    the full context history in its channels, so one token holds both.
    """
    config_class = Nucleus2MoEFMConcatConfig
    config_from_dict = staticmethod(_concat_config_from_dict)

    def __init__(self, config: Nucleus2MoEFMConcatConfig):
        super().__init__(config)
        if config.patching != "Linear":
            raise ValueError("Channel-concatenated conditioning only supports Linear patching")
        num_fields = len(self.expected_fields)
        self.embed = LinearEmbed(
            patch_size=config.patch_size,
            in_channels=num_fields * (1 + config.num_context_frames)
            + (config.num_condition_channels if self.concatenates_condition else 0),
            embed_dim=config.embed_dim,
            dtype=self.embed_dtype,
        )

    def _network_frames(self, input: torch.Tensor, num_context_frames: int,
                        condition: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        if num_context_frames != self.config.num_context_frames:
            raise ValueError(
                f"model was built for {self.config.num_context_frames} context frames, got {num_context_frames}"
            )
        context = input[:, :num_context_frames]
        noisy_frames = input[:, num_context_frames:]
        context_channels = einops.rearrange(context, "b t h w c -> b 1 h w (t c)")
        context_channels = context_channels.expand(-1, noisy_frames.shape[1], -1, -1, -1)
        channels = [noisy_frames, context_channels]
        if condition is not None:
            channels.append(condition[:, None].expand(-1, noisy_frames.shape[1], -1, -1, -1))
        return torch.cat(channels, dim=-1), 0
