"""Sinusoidal diffusion/progress embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_period = int(max_period)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=value.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = value.float()[..., None] * frequencies
        output = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if self.dim % 2:
            output = torch.nn.functional.pad(output, (0, 1))
        return output


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(timestep))
