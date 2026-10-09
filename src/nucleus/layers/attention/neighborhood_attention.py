import torch
import torch.nn as nn
from rotary_embedding_torch import apply_rotary_emb
import natten

from .natten_configs import get_natten_config

class NeighborhoodAttention(nn.Module):
    r"""
    This is similar to natten's NaighborhoodAttention2D,
    but includes additional query and key normalization.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        kernel_size: int = 3,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size

        assert self.head_dim % 16 == 0

        self.input_head = nn.Linear(embed_dim, 3 * embed_dim, dtype=dtype, bias=False)
        self.output_head = nn.Linear(embed_dim, embed_dim, dtype=dtype, bias=False)
        self.qnorm = nn.RMSNorm(self.head_dim, dtype=dtype)
        self.knorm = nn.RMSNorm(self.head_dim, dtype=dtype)
                
        natten.use_kv_parallelism_in_fused_na(mode=True)
        # Unrestricted may increase memory usage, but allows for better performance.
        natten.set_memory_usage_preference(pref='unrestricted')

    def forward(self, x, freqs):
        b, t, h, w, _ = x.shape

        heads = self.input_head(x).view(b, t, h, w, self.num_heads, 3 * self.head_dim)

        q, k, v = heads.tensor_split(3, dim=-1)
        q = self.qnorm(q)
        k = self.knorm(k)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # NATTEN CUDA kernels reject bf16 on pre-Ampere (sm < 80). Fall back to fp16
        # for the kernel call, then cast the output back to the module's dtype.
        kernel_dtype = q.dtype
        if (q.is_cuda and q.dtype == torch.bfloat16
                and torch.cuda.get_device_capability(q.device)[0] < 8):
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        output = natten.functional.na3d(
            q,
            k,
            v,
            kernel_size=(t, self.kernel_size, self.kernel_size),
            dilation=1,
            **get_natten_config(x.device)
        )

        output = output.to(kernel_dtype).view(b, t, h, w, self.num_heads * self.head_dim)
        output = self.output_head(output)
        return output