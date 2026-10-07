import torch
import torch.nn as nn
from torch.profiler import record_function

from nucleus.data.batching import CollatedBatch
from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.models._api import register_model
from nucleus.models.nucleus2_moe import Nucleus2MoE, Nucleus2MoEConfig, get_dtype


@register_model("nucleus2_moe_fm")
class Nucleus2MoEFlowMatching(Nucleus2MoE):
    """
    Nucleus2MoE with a dedicated per-block FiLM layer for flow time t.
    """

    def __init__(self, config: Nucleus2MoEConfig):
        super().__init__(config)
        self.t_norms = nn.ModuleList([
            AdaptiveLayerNorm(config.embed_dim, 1, dtype=get_dtype(config.activation_dtype))
            for _ in range(config.processor_blocks)
        ])

    def forward(self, batch: CollatedBatch):
        raise NotImplementedError("Call step(input, sim_params, t) directly.")

    def step(self, input: torch.Tensor, sim_params: torch.Tensor, t: torch.Tensor, T_in: int = 0):
        """
        input:     (B, T_in + T_out, H, W, C)
        sim_params:(B, 13)
        t:         (B, 1)  flow time in [0, 1]
        T_in:      number of context frames to drop before debed — output is (B, T_out, H, W, C)
        zt:        (1-t)z0 + ty_n
        loss:      L(model(concat(y_n-1, zt), t), y_n - z0)
        """
        assert input.dtype == torch.float32
        assert sim_params.dtype == torch.float32
        assert t.dtype == torch.float32

        _, _, h, w, _ = input.shape

        with record_function("encode"):
            x = embed = self.embed(input.to(self.embed_dtype))

        x = x.to(get_dtype(self.config.activation_dtype))

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(
                    embed_t, embed_h, embed_w
                )[None, :, :, :, None, :]

        moe_outputs = []
        for idx, (blk, t_norm) in enumerate(zip(self.blocks, self.t_norms)):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                x = t_norm(x, t)
                moe_outputs.append(moe_output)

        with record_function("debed"):
            x = self.out_norm(x.to(self.debed_dtype) + embed.to(self.debed_dtype))
            x = self.debed(x[:, T_in:, ...], target_shape=(h, w))

        return x.to(torch.float32), moe_outputs
