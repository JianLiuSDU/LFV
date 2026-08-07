"""Stable data contract for Stage 2 functional motion."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


ARRAY_SHAPES = {
    "manipulated_points": (256, 3),
    "reference_points": (256, 3),
    "goal_pose9d": (9,),
    "trajectory_pose9d": (64, 9),
}


def validate_functional_motion_sample(sample: Mapping[str, Any]) -> None:
    required = (
        "manipulated_points",
        "manipulated_dino",
        "reference_points",
        "reference_dino",
        "goal_pose9d",
        "trajectory_pose9d",
        "episode_id",
        "object_instance_id",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Missing Stage 2 fields: {missing}")
    for key, shape in ARRAY_SHAPES.items():
        if tuple(sample[key].shape) != shape:
            raise ValueError(f"{key} expected {shape}, got {tuple(sample[key].shape)}")
    for prefix in ("manipulated", "reference"):
        points = sample[f"{prefix}_points"]
        dino = sample[f"{prefix}_dino"]
        if dino.ndim != 2 or dino.shape[0] != points.shape[0]:
            raise ValueError(
                f"{prefix} DINO must be [256,D] and align with points, got {dino.shape}"
            )
    for key in required[:6]:
        value = sample[key]
        finite = torch.isfinite(value).all() if torch.is_tensor(value) else np.isfinite(value).all()
        if not bool(finite):
            raise ValueError(f"{key} contains NaN or Inf")
    for key in ("episode_id", "object_instance_id"):
        if not isinstance(sample[key], str) or not sample[key]:
            raise ValueError(f"{key} must be a non-empty string")
