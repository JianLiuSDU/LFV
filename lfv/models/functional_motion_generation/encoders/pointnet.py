"""Minimal role-specific PointNet branch."""

from __future__ import annotations

import torch
from torch import nn


class PointNetBranch(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        point_features = self.network(features)
        global_features = point_features.max(dim=1).values
        return point_features, global_features
