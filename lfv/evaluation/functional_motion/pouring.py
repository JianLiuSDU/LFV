"""Task-level pouring geometry metrics.

The functions in this module deliberately take explicit rim/opening geometry.
They never infer a success label from the cup center, which was the source of
ambiguous conclusions in the earlier simulation reports.
"""

from __future__ import annotations

import numpy as np


def _normalize(vector: np.ndarray, *, axis: int = -1) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector, axis=axis, keepdims=True).clip(min=1e-8)


def rim_over_opening_fraction(
    rim_points_world: np.ndarray,
    opening_center_world: np.ndarray,
    opening_normal_world: np.ndarray,
    opening_radius_m: float,
    *,
    height_tolerance_m: float = 0.02,
) -> float:
    """Return the fraction of rim samples above the circular bowl opening."""

    rim = np.asarray(rim_points_world, dtype=np.float32).reshape(-1, 3)
    if rim.shape[0] == 0:
        raise ValueError("rim_points_world must contain at least one point")
    center = np.asarray(opening_center_world, dtype=np.float32).reshape(3)
    normal = _normalize(np.asarray(opening_normal_world, dtype=np.float32).reshape(3))
    radius = float(opening_radius_m)
    if radius <= 0 or height_tolerance_m < 0:
        raise ValueError("opening_radius_m must be positive and tolerance non-negative")
    relative = rim - center
    signed_height = relative @ normal
    planar = relative - signed_height[:, None] * normal[None]
    inside = (np.linalg.norm(planar, axis=1) <= radius) & (
        np.abs(signed_height) <= float(height_tolerance_m)
    )
    return float(inside.mean())


def continuous_rim_arc_fraction(
    rim_points_world: np.ndarray,
    opening_center_world: np.ndarray,
    opening_normal_world: np.ndarray,
    opening_radius_m: float,
    *,
    height_tolerance_m: float = 0.02,
) -> float:
    """Return the longest circularly-contiguous in-opening rim arc fraction."""

    rim = np.asarray(rim_points_world, dtype=np.float32).reshape(-1, 3)
    center = np.asarray(opening_center_world, dtype=np.float32).reshape(3)
    normal = _normalize(np.asarray(opening_normal_world, dtype=np.float32).reshape(3))
    relative = rim - center
    signed_height = relative @ normal
    planar = relative - signed_height[:, None] * normal[None]
    inside = (np.linalg.norm(planar, axis=1) <= float(opening_radius_m)) & (
        np.abs(signed_height) <= float(height_tolerance_m)
    )
    if not inside.any():
        return 0.0
    # The rim samples need not be supplied in angular order.  Construct a
    # stable in-plane basis and sort only for this evaluation metric.
    basis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(basis @ normal)) > 0.9:
        basis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    basis = _normalize(basis - (basis @ normal) * normal)
    second = _normalize(np.cross(normal, basis))
    angles = np.arctan2(planar @ second, planar @ basis)
    order = np.argsort(angles)
    ordered = inside[order]
    doubled = np.concatenate((ordered, ordered))
    longest = current = 0
    for value in doubled:
        current = current + 1 if value else 0
        longest = max(longest, current)
    longest = min(longest, ordered.size)
    return float(longest / ordered.size)


def pouring_success(
    rim_points_world: np.ndarray,
    opening_center_world: np.ndarray,
    opening_normal_world: np.ndarray,
    opening_radius_m: float,
    *,
    min_rof: float = 0.20,
    height_tolerance_m: float = 0.02,
) -> dict[str, float | bool]:
    rof = rim_over_opening_fraction(
        rim_points_world,
        opening_center_world,
        opening_normal_world,
        opening_radius_m,
        height_tolerance_m=height_tolerance_m,
    )
    arc = continuous_rim_arc_fraction(
        rim_points_world,
        opening_center_world,
        opening_normal_world,
        opening_radius_m,
        height_tolerance_m=height_tolerance_m,
    )
    return {
        "rim_over_opening_fraction": rof,
        "continuous_rim_arc_fraction": arc,
        "success": bool(rof >= float(min_rof) and arc > 0.0),
    }

