"""Goal-conditioned scene mixing and ordered latent phase tokens."""

from __future__ import annotations

import torch
from torch import nn


class GoalConditionedContextMixer(nn.Module):
    """Mix the three scene tokens with the candidate goal before decoding."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        residual_gating: bool = False,
        residual_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.residual_gate = (
            nn.Parameter(torch.full((hidden_dim,), float(residual_gate_init)))
            if residual_gating
            else None
        )

    def forward(self, scene_tokens: torch.Tensor, goal_token: torch.Tensor) -> torch.Tensor:
        memory = torch.cat((scene_tokens, goal_token), dim=1)
        original = memory
        for layer in self.layers:
            memory = layer(memory)
        if self.residual_gate is not None:
            return original + self.residual_gate[None, None] * (memory - original)
        return self.output_norm(memory)


class LatentPhaseTokenGenerator(nn.Module):
    """Generate a small ordered set of task-phase tokens from scene and goal."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.queries = nn.Parameter(torch.empty(self.num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.queries[None].expand(memory.shape[0], -1, -1)
        normalized_memory = self.memory_norm(memory)
        update = self.cross_attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )[0]
        return self.output_norm(self.self_layer(queries + update))
