"""Single-token conditional Goal Pose transformer."""

from __future__ import annotations

import torch
from torch import nn

from ..blocks import GoalConditionBlock, TimestepEmbedding


class GoalPoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pose_embedding = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.timestep_embedding = TimestepEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                GoalConditionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 9)
        with torch.no_grad():
            self.output.bias.zero_()
            self.output.bias[3] = 1.0
            self.output.bias[7] = 1.0

    def forward(
        self,
        noisy_goal: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        goal_relation_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token = self.pose_embedding(noisy_goal)[:, None]
        time = self.timestep_embedding(timestep)
        if goal_relation_tokens is not None:
            context = torch.cat((context, goal_relation_tokens), dim=1)
        for block in self.blocks:
            token = block(token, context, time)
        return self.output(self.output_norm(token[:, 0]))
