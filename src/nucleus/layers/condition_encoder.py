import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encodes static (B, H, W, K) maps into patch-grid tokens (B, H/p, W/p, D) to add to field tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, hidden_channels: int = 32,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        # two full-resolution convolutions form nonlinear, slightly non-local features (e.g. "a site
        # is next to this patch") before the patch projection blurs them
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
        )
        self.project = nn.Conv2d(hidden_channels, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)
        # start as a no-op, so a conditioned model begins exactly like an unconditioned one
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        channels_first = maps.permute(0, 3, 1, 2)
        return self.project(self.features(channels_first)).permute(0, 2, 3, 1)
