from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn
from torch.profiler import record_function

from nucleus.data.batching import CollatedBatch
from nucleus.layers.fm_time_embed import FMTimeEmbed
from nucleus.layers.patching import LinearEmbed
from nucleus.models._api import register_model
from nucleus.models.nucleus2_moe import (
    Nucleus2MoE, Nucleus2MoEConfig, TransformerMoEBlock,
    _config_from_dict, get_dtype,
)


@register_model("nucleus2_moe_fm")
class Nucleus2MoEFlowMatching(Nucleus2MoE):
    """
    Nucleus2MoE adapted for flow matching.

    Context and noisy frames are stacked along time; each frame is its own token.

    t conditioning: the flow-time embedding is concatenated to sim_params before
    every block, so both the attention norm and the MoE norm inside each block see t
    directly.  This replaces the old post-block t_norms, which normalised the
    residual stream at each layer boundary (effectively post-norm) and hid t from
    the block internals.
    """

    def __init__(self, config: Nucleus2MoEConfig):
        super().__init__(config)
        self.time_embed = FMTimeEmbed(config.embed_dim)
        # Rebuild blocks: AdaptiveLayerNorm inside each block now takes
        # (num_sim_params + embed_dim) features so t is visible inside attention + MoE.
        # The base class built blocks with num_sim_params=13; replace them here.
        aug_num_sim_params = self.num_sim_params + config.embed_dim
        drop_path_probs = torch.linspace(0.0, 0.1, config.processor_blocks)
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                config=config,
                num_sim_params=aug_num_sim_params,
                drop_path_prob=drop_path_probs[idx].item(),
            )
            for idx in range(config.processor_blocks)
        ])

    def forward(self, batch: CollatedBatch):
        raise NotImplementedError("Call step(input, sim_params, t) directly.")

    def _network_frames(self, input: torch.Tensor, num_context_frames: int,
                        condition: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        # returns the frames fed to the network and how many leading frames to drop before decoding
        if condition is not None:
            input = torch.cat([input, condition[:, None].expand(-1, input.shape[1], -1, -1, -1)], dim=-1)
        return input, num_context_frames

    def _skip(self, frames: torch.Tensor, input: torch.Tensor, T_in: int,
              condition: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Embedding used for the residual skip connection before debedding.
        The default uses the full frame embedding (same as the transformer input).
        Subclasses can override to use only the noisy-frame channels.
        """
        return self._encode(frames, condition)

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
            skip = self._skip(frames, input, T_in, condition)

        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(
                    embed_t, embed_h, embed_w
                )[None, :, :, :, None, :]

        # Concat time embedding to sim_params so each block's AdaptiveLayerNorm
        # (both attention and MoE paths) is conditioned on t without touching the
        # residual stream between blocks.
        time_features = self.time_embed(t)  # (B, embed_dim)
        aug_sim_params = torch.cat([sim_params, time_features], dim=-1)  # (B, 13 + embed_dim)

        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, aug_sim_params)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + skip.to(self.debed_dtype))
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
    Flow matching with channel-concatenated conditioning: each noisy frame is paired
    with the full context history in its channels, so one token holds both.

    The skip connection uses a separate zt-only embedding so context channels don't
    contaminate the velocity residual that the debed must invert.
    """
    config_class = Nucleus2MoEFMConcatConfig
    config_from_dict = staticmethod(_concat_config_from_dict)

    def __init__(self, config: Nucleus2MoEFMConcatConfig):
        super().__init__(config)
        if config.patching != "Linear":
            raise ValueError("Channel-concatenated conditioning only supports Linear patching")
        num_fields = len(self.expected_fields)
        # Main embed: noisy frame + all context channels (+ optional condition)
        self.embed = LinearEmbed(
            patch_size=config.patch_size,
            in_channels=num_fields * (1 + config.num_context_frames)
            + (config.num_condition_channels if self.concatenates_condition else 0),
            embed_dim=config.embed_dim,
            dtype=self.embed_dtype,
        )
        # Skip embed: noisy frame only — context channels don't belong in the
        # velocity/endpoint residual that the debed has to decode.
        self.zt_skip_embed = LinearEmbed(
            patch_size=config.patch_size,
            in_channels=num_fields,
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

    def _skip(self, frames: torch.Tensor, input: torch.Tensor, T_in: int,
              condition: Optional[torch.Tensor]) -> torch.Tensor:
        noisy_frames = input[:, T_in:]
        return self.zt_skip_embed(noisy_frames.to(self.embed_dtype))
