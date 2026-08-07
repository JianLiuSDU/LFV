"""Timestep-conditioned pre-normalization."""

from __future__ import annotations

import torch
from torch import nn


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2)
        )

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(tokens) * (1.0 + scale[:, None]) + shift[:, None]
