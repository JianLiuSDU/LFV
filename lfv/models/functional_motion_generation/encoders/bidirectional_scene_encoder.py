"""Two PointNets and two directional cross-attention relations."""

from __future__ import annotations

import torch
from torch import nn

from ..interfaces import ContextEncoding
from .pointnet import PointNetBranch


class DirectionalRelation(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=True,
            average_attn_weights=False,
        )
        relation_points = self.fusion(torch.cat((query, attended), dim=-1))
        return relation_points.max(dim=1).values, weights


class BidirectionalSceneEncoder(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        dino_projected_dim: int = 64,
        xyz_projected_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dino_projected_dim + xyz_projected_dim != hidden_dim:
            raise ValueError("DINO and XYZ projected dimensions must sum to hidden_dim")
        self.dino_projector = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, 256),
            nn.GELU(),
            nn.Linear(256, dino_projected_dim),
        )
        self.manipulated_xyz = nn.Sequential(
            nn.Linear(3, xyz_projected_dim), nn.GELU()
        )
        self.reference_xyz = nn.Sequential(
            nn.Linear(3, xyz_projected_dim), nn.GELU()
        )
        self.manipulated_pointnet = PointNetBranch(hidden_dim, hidden_dim)
        self.reference_pointnet = PointNetBranch(hidden_dim, hidden_dim)
        self.initial_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.manipulated_queries_reference = DirectionalRelation(
            hidden_dim, num_heads, dropout
        )
        self.reference_queries_manipulated = DirectionalRelation(
            hidden_dim, num_heads, dropout
        )
        self.type_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)

    def forward(
        self,
        manipulated_points: torch.Tensor,
        manipulated_dino: torch.Tensor,
        reference_points: torch.Tensor,
        reference_dino: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> ContextEncoding:
        manipulated_input = torch.cat(
            (self.manipulated_xyz(manipulated_points), self.dino_projector(manipulated_dino)),
            dim=-1,
        )
        reference_input = torch.cat(
            (self.reference_xyz(reference_points), self.dino_projector(reference_dino)),
            dim=-1,
        )
        manipulated_features, manipulated_global = self.manipulated_pointnet(
            manipulated_input
        )
        reference_features, reference_global = self.reference_pointnet(reference_input)
        initial = self.initial_fusion(
            torch.cat((manipulated_global, reference_global), dim=-1)
        )
        manipulated_relation, attention_mr = self.manipulated_queries_reference(
            manipulated_features, reference_features
        )
        reference_relation, attention_rm = self.reference_queries_manipulated(
            reference_features, manipulated_features
        )
        tokens = torch.stack(
            (initial, manipulated_relation, reference_relation), dim=1
        ) + self.type_embedding[None]
        if not return_debug:
            return ContextEncoding(tokens=tokens)
        reference_importance = attention_mr.mean(dim=(1, 2))
        manipulated_importance = attention_rm.mean(dim=(1, 2))
        return ContextEncoding(
            tokens=tokens,
            attention_manipulated_to_reference=attention_mr,
            attention_reference_to_manipulated=attention_rm,
            reference_importance=reference_importance,
            manipulated_importance=manipulated_importance,
        )
