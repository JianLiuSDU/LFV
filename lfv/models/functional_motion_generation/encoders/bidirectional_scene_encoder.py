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
    if intervention == "drop_top":
        # Remove the highest-mass entries and renormalize.  This is used only
        # as a no-gradient counterfactual probe: if the learned field is
        # task-relevant, suppressing its peak should increase the denoising
        # loss relative to the learned distribution.
        flat = distribution.flatten(1)
        num_drop = max(1, int(round(0.1 * flat.shape[1])))
        top_indices = torch.topk(flat, k=num_drop, dim=1).indices
        output = flat.scatter(
            1,
            top_indices,
            torch.zeros_like(top_indices, dtype=flat.dtype),
        )
        return (output / output.sum(dim=1, keepdim=True).clamp_min(1e-8)).reshape_as(
            distribution
        )
    raise ValueError("motion_field_intervention must be None, 'uniform', or 'roll'")


def _sharpen_distribution(
    distribution: torch.Tensor,
    power: float,
) -> torch.Tensor:
    """Sharpen a learned field without introducing coordinates or masks."""

    if power <= 0:
        raise ValueError("motion_field_power must be positive")
    if abs(float(power) - 1.0) < 1e-8:
        return distribution
    axes = tuple(range(1, distribution.ndim))
    # Work in log space.  Joint pair fields can contain 1/(N*M) masses; a
    # direct fourth power would underflow to zero before normalization.
    log_values = distribution.clamp_min(1e-12).log() * float(power)
    normalizer = torch.logsumexp(log_values.flatten(1), dim=1).reshape(
        (distribution.shape[0],) + (1,) * (distribution.ndim - 1)
    )
    return (log_values - normalizer).exp()


def _mix_field(
    current: torch.Tensor,
    prior: torch.Tensor,
    weight: float,
    *,
    mode: str = "fixed",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse two per-point relevance distributions.

    ``fixed`` preserves the original arithmetic interpolation.  ``confidence``
    additionally attenuates the requested prior weight when the online and
    transported distributions disagree (a Jensen--Shannon evidence check).
    The returned second tensor is the effective per-example prior weight and is
    exposed in ``ContextEncoding`` for diagnostics.
    """

    if current.shape != prior.shape:
        raise ValueError(f"Motion-field prior shape {tuple(prior.shape)} != current {tuple(current.shape)}")
    prior = prior.to(device=current.device, dtype=current.dtype).clamp_min(0.0)
    current = current.clamp_min(0.0)
    current = current / current.sum(dim=1, keepdim=True).clamp_min(1e-8)
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if mode not in {"fixed", "confidence"}:
        raise ValueError("motion_field_fusion_mode must be 'fixed' or 'confidence'")
    strength = torch.full(
        (current.shape[0], 1),
        float(max(0.0, min(1.0, weight))),
        device=current.device,
        dtype=current.dtype,
    )
    if mode == "confidence":
        midpoint = 0.5 * (current + prior)
        js = 0.5 * (
            (current * (current.clamp_min(1e-8) / midpoint.clamp_min(1e-8)).log()).sum(dim=1)
            + (prior * (prior.clamp_min(1e-8) / midpoint.clamp_min(1e-8)).log()).sum(dim=1)
        )
        # JS is bounded by log(2); map agreement to [0, 1] without a learned
        # gate or a task-specific geometric heuristic.
        agreement = (1.0 - js / torch.log(torch.as_tensor(2.0, device=current.device, dtype=current.dtype))).clamp(0.0, 1.0)
        strength = strength * agreement[:, None]
    mixed = (1.0 - strength) * current + strength * prior
    return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8), strength


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
        motion_field_power: float = 1.0,
    ) -> None:
        super().__init__()
        if motion_field_mode not in {"none", "independent", "joint"}:
            raise ValueError(
                "motion_field_mode must be one of {'none', 'independent', 'joint'}"
            )
        if motion_field_temperature <= 0:
            raise ValueError("motion_field_temperature must be positive")
        if motion_field_power <= 0:
            raise ValueError("motion_field_power must be positive")
        self.motion_field_mode = motion_field_mode
        self.motion_field_temperature = float(motion_field_temperature)
        self.motion_field_power = float(motion_field_power)
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
            field = _sharpen_distribution(
                torch.softmax(logits / self.motion_field_temperature, dim=1),
                self.motion_field_power,
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
        motion_field_power: float = 1.0,
        motion_field_pair_weight: float = 0.25,
        motion_field_fusion_mode: str = "fixed",
        motion_field_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        if dino_projected_dim + xyz_projected_dim != hidden_dim:
            raise ValueError("DINO and XYZ projected dimensions must sum to hidden_dim")
        if motion_field_pair_weight < 0:
            raise ValueError("motion_field_pair_weight must be non-negative")
        self.motion_field_mode = motion_field_mode
        self.motion_field_temperature = float(motion_field_temperature)
        if motion_field_power <= 0:
            raise ValueError("motion_field_power must be positive")
        self.motion_field_power = float(motion_field_power)
        self.motion_field_pair_weight = float(motion_field_pair_weight)
        if motion_field_fusion_mode not in {"fixed", "confidence"}:
            raise ValueError("motion_field_fusion_mode must be 'fixed' or 'confidence'")
        self.motion_field_fusion_mode = str(motion_field_fusion_mode)
        self.motion_field_bottleneck = bool(motion_field_bottleneck)
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
            motion_field_power=motion_field_power,
        )
        self.reference_queries_manipulated = DirectionalRelation(
            hidden_dim,
            num_heads,
            dropout,
            motion_field_mode=motion_field_mode,
            motion_field_temperature=motion_field_temperature,
            motion_field_power=motion_field_power,
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
        motion_field_intervention: str | None = None,
        motion_field_prior: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        motion_field_prior_weight: float = 0.0,
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
        fusion_weights = None
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
            joint_relation = torch.softmax(
                joint_logits.flatten(1) / self.motion_field_temperature,
                dim=1,
            ).reshape_as(joint_logits)
            joint_relation = _sharpen_distribution(
                joint_relation, self.motion_field_power
            )
            joint_relation = _intervene_distribution(
                joint_relation, motion_field_intervention
            )
            manipulated_field = joint_relation.sum(dim=2)
            reference_field = joint_relation.sum(dim=1)
            if motion_field_prior is not None and motion_field_prior_weight > 0.0:
                prior_m, prior_r = motion_field_prior
                if prior_m is not None and prior_r is not None:
                    manipulated_field, weight_m = _mix_field(
                        manipulated_field,
                        prior_m,
                        motion_field_prior_weight,
                        mode=self.motion_field_fusion_mode,
                    )
                    reference_field, weight_r = _mix_field(
                        reference_field,
                        prior_r,
                        motion_field_prior_weight,
                        mode=self.motion_field_fusion_mode,
                    )
                    fusion_weights = 0.5 * (weight_m + weight_r)
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
            if (
                self.motion_field_mode == "independent"
                and motion_field_prior is not None
                and motion_field_prior_weight > 0.0
            ):
                prior_m, prior_r = motion_field_prior
                if prior_m is not None and prior_r is not None:
                    manipulated_relation.motion_field, weight_m = _mix_field(
                        manipulated_relation.motion_field,
                        prior_m,
                        motion_field_prior_weight,
                        mode=self.motion_field_fusion_mode,
                    )
                    reference_relation.motion_field, weight_r = _mix_field(
                        reference_relation.motion_field,
                        prior_r,
                        motion_field_prior_weight,
                        mode=self.motion_field_fusion_mode,
                    )
                    fusion_weights = 0.5 * (weight_m + weight_r)
                    manipulated_relation.token = torch.sum(
                        manipulated_relation.motion_field.unsqueeze(-1) * manipulated_relation.points,
                        dim=1,
                    )
                    reference_relation.token = torch.sum(
                        reference_relation.motion_field.unsqueeze(-1) * reference_relation.points,
                        dim=1,
                    )
            if (
                self.motion_field_bottleneck
                and manipulated_relation.motion_field is not None
                and reference_relation.motion_field is not None
            ):
                manipulated_summary = torch.sum(
                    manipulated_relation.motion_field.unsqueeze(-1)
                    * manipulated_relation.points,
                    dim=1,
                )
                reference_summary = torch.sum(
                    reference_relation.motion_field.unsqueeze(-1)
                    * reference_relation.points,
                    dim=1,
                )
                initial = self.initial_fusion(
                    torch.cat((manipulated_summary, reference_summary), dim=-1)
                )
            else:
                initial = self.initial_fusion(
                    torch.cat((manipulated_global, reference_global), dim=-1)
                )
        tokens = torch.stack(
            (initial, manipulated_relation.token, reference_relation.token), dim=1
        ) + self.type_embedding[None]
        if not return_debug:
            return ContextEncoding(
                tokens=tokens,
                manipulated_motion_field=manipulated_relation.motion_field,
                reference_motion_field=reference_relation.motion_field,
                manipulated_motion_logits=manipulated_relation.motion_logits,
                reference_motion_logits=reference_relation.motion_logits,
                joint_motion_relation=joint_relation,
                motion_field_fusion_weight=fusion_weights,
            )
        attention_mr = manipulated_relation.attention
        attention_rm = reference_relation.attention
        reference_importance = attention_mr.mean(dim=(1, 2))
        manipulated_importance = attention_rm.mean(dim=(1, 2))
        return ContextEncoding(
            tokens=tokens,
            manipulated_motion_field=manipulated_relation.motion_field,
            reference_motion_field=reference_relation.motion_field,
            manipulated_motion_logits=manipulated_relation.motion_logits,
            reference_motion_logits=reference_relation.motion_logits,
            joint_motion_relation=joint_relation,
            attention_manipulated_to_reference=attention_mr,
            attention_reference_to_manipulated=attention_rm,
            reference_importance=reference_importance,
            manipulated_importance=manipulated_importance,
            motion_field_fusion_weight=fusion_weights,
        )
