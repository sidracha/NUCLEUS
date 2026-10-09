import torch
import torch.nn as nn


class FMTimeEmbed(nn.Module):
    def __init__(self, embed_dim: int, num_frequencies: int = 32, max_period: float = 10000.0):
        super().__init__()
        exponents = torch.arange(num_frequencies, dtype=torch.float32) / num_frequencies
        self.register_buffer("frequencies", max_period ** (-exponents), persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        # flow time lives in [0, 1]; scale it up so the low frequencies still turn
        # noticeably along the path, matching the usual diffusion-step convention
        angles = 1000.0 * flow_time * self.frequencies
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(features)
