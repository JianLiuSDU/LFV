"""Typed outputs shared by Stage 2 models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ContextEncoding:
    tokens: torch.Tensor
    manipulated_motion_field: torch.Tensor | None = None
    reference_motion_field: torch.Tensor | None = None
    manipulated_motion_logits: torch.Tensor | None = None
    reference_motion_logits: torch.Tensor | None = None
    joint_motion_relation: torch.Tensor | None = None
    attention_manipulated_to_reference: torch.Tensor | None = None
    attention_reference_to_manipulated: torch.Tensor | None = None
    reference_importance: torch.Tensor | None = None
    manipulated_importance: torch.Tensor | None = None
    motion_field_fusion_weight: torch.Tensor | None = None


@dataclass
class Stage2Samples:
    goals: torch.Tensor
    trajectories: torch.Tensor
    goal_ids: torch.Tensor
