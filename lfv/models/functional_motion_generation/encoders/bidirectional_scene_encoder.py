"""Two PointNets and two directional cross-attention relations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..interfaces import ContextEncoding
from .pointnet import PointNetBranch


def _intervene_distribution(
    distribution: torch.Tensor,
    intervention: str | None,
) -> torch.Tensor:
    if intervention is None:
        return distribution
    if intervention == "uniform":
        return torch.full_like(distribution, 1.0 / distribution[0].numel())
    if intervention == "roll":
        output = distribution
        for dimension in range(1, distribution.ndim):
            output = torch.roll(
                output,
                shifts=distribution.shape[dimension] // 2,
                dims=dimension,
            )
        return output
    raise ValueError("motion_field_intervention must be None, 'uniform', or 'roll'")


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection onto the probability simplex.

    Unlike softmax, sparsemax can assign exact zero mass to irrelevant points,
    which makes the learned field easier to inspect without adding a spatial
    heuristic.  The implementation follows the standard sorting formulation.
    """

    values = logits.transpose(dim, -1)
    sorted_values, _ = torch.sort(values, dim=-1, descending=True)
    cssv = sorted_values.cumsum(dim=-1) - 1
    positions = torch.arange(
        1,
        values.shape[-1] + 1,
        device=values.device,
        dtype=values.dtype,
    )
    support = positions * sorted_values > cssv
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = cssv.gather(-1, support_size.long() - 1) / support_size
    output = torch.clamp(values - tau, min=0)
    return output.transpose(dim, -1)


def _field_distribution(
    logits: torch.Tensor,
    temperature: float,
    normalization: str,
    dim: int,
) -> torch.Tensor:
    scaled = logits / temperature
    if normalization == "softmax":
        return torch.softmax(scaled, dim=dim)
    if normalization == "sparsemax":
        return _sparsemax(scaled, dim=dim)
    raise ValueError("motion_field_normalization must be 'softmax' or 'sparsemax'")


@dataclass
class DirectionalRelationEncoding:
    token: torch.Tensor
    points: torch.Tensor
    attention: torch.Tensor
    motion_logits: torch.Tensor | None = None
    motion_field: torch.Tensor | None = None


class DirectionalRelation(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        *,
        motion_field_mode: str = "none",
        motion_field_temperature: float = 1.0,
        motion_field_normalization: str = "softmax",
    ) -> None:
        super().__init__()
        if motion_field_mode not in {"none", "independent", "joint"}:
            raise ValueError(
                "motion_field_mode must be one of {'none', 'independent', 'joint'}"
            )
        if motion_field_temperature <= 0:
            raise ValueError("motion_field_temperature must be positive")
        self.motion_field_mode = motion_field_mode
        self.motion_field_temperature = float(motion_field_temperature)
        if motion_field_normalization not in {"softmax", "sparsemax"}:
            raise ValueError("motion_field_normalization must be 'softmax' or 'sparsemax'")
        self.motion_field_normalization = motion_field_normalization
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
        self.relevance_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if motion_field_mode != "none"
            else None
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> DirectionalRelationEncoding:
        attended, weights = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=True,
            average_attn_weights=False,
        )
        relation_points = self.fusion(torch.cat((query, attended), dim=-1))
        if self.relevance_head is None:
            return DirectionalRelationEncoding(
                token=relation_points.max(dim=1).values,
                points=relation_points,
                attention=weights,
            )
        logits = self.relevance_head(relation_points).squeeze(-1)
        field = None
        token = relation_points.max(dim=1).values
        if self.motion_field_mode == "independent":
            field = _field_distribution(
                logits,
                self.motion_field_temperature,
                self.motion_field_normalization,
                dim=1,
            )
            token = torch.sum(field.unsqueeze(-1) * relation_points, dim=1)
        return DirectionalRelationEncoding(
            token=token,
            points=relation_points,
            attention=weights,
            motion_logits=logits,
            motion_field=field,
        )


class BidirectionalSceneEncoder(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        dino_projected_dim: int = 64,
        xyz_projected_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        motion_field_mode: str = "none",
        motion_field_temperature: float = 1.0,
        motion_field_normalization: str = "softmax",
        motion_field_pair_weight: float = 0.25,
        goal_relation_conditioning: bool = False,
        goal_relation_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if dino_projected_dim + xyz_projected_dim != hidden_dim:
            raise ValueError("DINO and XYZ projected dimensions must sum to hidden_dim")
        if motion_field_pair_weight < 0:
            raise ValueError("motion_field_pair_weight must be non-negative")
        self.motion_field_mode = motion_field_mode
        self.motion_field_temperature = float(motion_field_temperature)
        self.motion_field_normalization = motion_field_normalization
        self.motion_field_pair_weight = float(motion_field_pair_weight)
        self.goal_relation_conditioning = bool(goal_relation_conditioning)
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
            hidden_dim,
            num_heads,
            dropout,
            motion_field_mode=motion_field_mode,
            motion_field_temperature=motion_field_temperature,
            motion_field_normalization=motion_field_normalization,
        )
        self.reference_queries_manipulated = DirectionalRelation(
            hidden_dim,
            num_heads,
            dropout,
            motion_field_mode=motion_field_mode,
            motion_field_temperature=motion_field_temperature,
            motion_field_normalization=motion_field_normalization,
        )
        self.type_embedding = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        if self.goal_relation_conditioning:
            # The anchor is a differentiable moment of the learned field, not
            # a hand-coded centroid or an OBB.  Separate role projections keep
            # the object roles identifiable while sharing the same geometry.
            self.manipulated_anchor_encoder = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.reference_anchor_encoder = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.goal_relation_encoder = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            # Start as a small residual branch so a newly added relation
            # condition cannot destroy the well-tested three-token baseline.
            self.goal_relation_gate = nn.Parameter(
                torch.tensor(float(goal_relation_gate_init))
            )

    def forward(
        self,
        manipulated_points: torch.Tensor,
        manipulated_dino: torch.Tensor,
        reference_points: torch.Tensor,
        reference_dino: torch.Tensor,
        *,
        return_debug: bool = False,
        motion_field_intervention: str | None = None,
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
        manipulated_relation = self.manipulated_queries_reference(
            manipulated_features, reference_features
        )
        reference_relation = self.reference_queries_manipulated(
            reference_features, manipulated_features
        )
        joint_relation = None
        if self.motion_field_mode == "joint":
            if (
                manipulated_relation.motion_logits is None
                or reference_relation.motion_logits is None
            ):
                raise RuntimeError("Joint motion field requires relevance logits")
            attention_mr = manipulated_relation.attention.mean(dim=1)
            attention_rm = reference_relation.attention.mean(dim=1).transpose(1, 2)
            pair_log_compatibility = 0.5 * (
                attention_mr.clamp_min(1e-8).log()
                + attention_rm.clamp_min(1e-8).log()
            )
            joint_logits = (
                manipulated_relation.motion_logits.unsqueeze(2)
                + reference_relation.motion_logits.unsqueeze(1)
                + self.motion_field_pair_weight * pair_log_compatibility
            )
            joint_relation = _field_distribution(
                joint_logits.flatten(1),
                self.motion_field_temperature,
                self.motion_field_normalization,
                dim=1,
            ).reshape_as(joint_logits)
            joint_relation = _intervene_distribution(
                joint_relation, motion_field_intervention
            )
            manipulated_field = joint_relation.sum(dim=2)
            reference_field = joint_relation.sum(dim=1)
            manipulated_relation.motion_field = manipulated_field
            reference_relation.motion_field = reference_field
            manipulated_relation.token = torch.sum(
                manipulated_field.unsqueeze(-1) * manipulated_relation.points,
                dim=1,
            )
            reference_relation.token = torch.sum(
                reference_field.unsqueeze(-1) * reference_relation.points,
                dim=1,
            )
            manipulated_summary = torch.sum(
                manipulated_field.unsqueeze(-1) * manipulated_features,
                dim=1,
            )
            reference_summary = torch.sum(
                reference_field.unsqueeze(-1) * reference_features,
                dim=1,
            )
            initial = self.initial_fusion(
                torch.cat((manipulated_summary, reference_summary), dim=-1)
            )
        else:
            if (
                self.motion_field_mode == "independent"
                and motion_field_intervention is not None
            ):
                manipulated_relation.motion_field = _intervene_distribution(
                    manipulated_relation.motion_field, motion_field_intervention
                )
                reference_relation.motion_field = _intervene_distribution(
                    reference_relation.motion_field, motion_field_intervention
                )
                manipulated_relation.token = torch.sum(
                    manipulated_relation.motion_field.unsqueeze(-1)
                    * manipulated_relation.points,
                    dim=1,
                )
                reference_relation.token = torch.sum(
                    reference_relation.motion_field.unsqueeze(-1)
                    * reference_relation.points,
                    dim=1,
                )
            initial = self.initial_fusion(
                torch.cat((manipulated_global, reference_global), dim=-1)
            )
        tokens = torch.stack(
            (initial, manipulated_relation.token, reference_relation.token), dim=1
        ) + self.type_embedding[None]
        goal_relation_tokens = None
        manipulated_anchor_xyz = None
        reference_anchor_xyz = None
        if self.goal_relation_conditioning:
            # A missing field (e.g. a legacy ``motion_field_mode=none`` model)
            # falls back to a uniform distribution.  This keeps the optional
            # conditioning well-defined without introducing a hard-coded point.
            if manipulated_relation.motion_field is None:
                manipulated_weights = torch.full(
                    manipulated_points.shape[:2],
                    1.0 / manipulated_points.shape[1],
                    device=manipulated_points.device,
                    dtype=manipulated_points.dtype,
                )
            else:
                manipulated_weights = manipulated_relation.motion_field
            if reference_relation.motion_field is None:
                reference_weights = torch.full(
                    reference_points.shape[:2],
                    1.0 / reference_points.shape[1],
                    device=reference_points.device,
                    dtype=reference_points.dtype,
                )
            else:
                reference_weights = reference_relation.motion_field
            manipulated_anchor_xyz = torch.sum(
                manipulated_weights.unsqueeze(-1) * manipulated_points, dim=1
            )
            reference_anchor_xyz = torch.sum(
                reference_weights.unsqueeze(-1) * reference_points, dim=1
            )
            manipulated_anchor = self.manipulated_anchor_encoder(
                torch.cat((manipulated_relation.token, manipulated_anchor_xyz), dim=-1)
            )
            reference_anchor = self.reference_anchor_encoder(
                torch.cat((reference_relation.token, reference_anchor_xyz), dim=-1)
            )
            relative_xyz = reference_anchor_xyz - manipulated_anchor_xyz
            relation_anchor = self.goal_relation_encoder(
                torch.cat((manipulated_anchor, reference_anchor, relative_xyz), dim=-1)
            )
            goal_relation_tokens = torch.stack(
                (manipulated_anchor, reference_anchor, relation_anchor), dim=1
            ) * self.goal_relation_gate
        if not return_debug:
            return ContextEncoding(
                tokens=tokens,
                goal_relation_tokens=goal_relation_tokens,
                manipulated_anchor_xyz=manipulated_anchor_xyz,
                reference_anchor_xyz=reference_anchor_xyz,
                manipulated_motion_field=manipulated_relation.motion_field,
                reference_motion_field=reference_relation.motion_field,
                manipulated_motion_logits=manipulated_relation.motion_logits,
                reference_motion_logits=reference_relation.motion_logits,
                joint_motion_relation=joint_relation,
            )
        attention_mr = manipulated_relation.attention
        attention_rm = reference_relation.attention
        reference_importance = attention_mr.mean(dim=(1, 2))
        manipulated_importance = attention_rm.mean(dim=(1, 2))
        return ContextEncoding(
            tokens=tokens,
            goal_relation_tokens=goal_relation_tokens,
            manipulated_anchor_xyz=manipulated_anchor_xyz,
            reference_anchor_xyz=reference_anchor_xyz,
            manipulated_motion_field=manipulated_relation.motion_field,
            reference_motion_field=reference_relation.motion_field,
            manipulated_motion_logits=manipulated_relation.motion_logits,
            reference_motion_logits=reference_relation.motion_logits,
            joint_motion_relation=joint_relation,
            attention_manipulated_to_reference=attention_mr,
            attention_reference_to_manipulated=attention_rm,
            reference_importance=reference_importance,
            manipulated_importance=manipulated_importance,
        )
