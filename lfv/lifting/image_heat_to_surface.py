from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiftedImageHeat:
    """Visible target surface reconstructed from one aligned RGB-D image."""

    pixels_uv: np.ndarray
    points_camera: np.ndarray
    heat: np.ndarray
    raw_heat: np.ndarray

    def summary(self) -> dict[str, float | int | list[float]]:
        active = self.heat > 0
        if np.any(active):
            weights = self.heat[active].astype(np.float64)
            center = np.average(
                self.points_camera[active], axis=0, weights=weights
            )
        else:
            center = np.full(3, np.nan, dtype=np.float64)
        return {
            "num_visible_points": int(len(self.points_camera)),
            "num_active_heat_points": int(active.sum()),
            "raw_heat_max": float(self.raw_heat.max(initial=0.0)),
            "thresholded_heat_max": float(self.heat.max(initial=0.0)),
            "active_heat_center_camera": center.astype(float).tolist(),
        }


def lift_image_heat_to_camera(
    heatmap: np.ndarray,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    intrinsic_cv: np.ndarray,
    *,
    heat_threshold: float = 0.15,
    minimum_depth_m: float = 1e-4,
    maximum_depth_m: float = 2.0,
) -> LiftedImageHeat:
    """Back-project an aligned image heatmap into the OpenCV camera frame.

    All valid object pixels are retained as visible geometry so hidden-surface
    checks can distinguish observed and unobserved surface.  Values below the
    task heat threshold are set to zero but remain present geometrically.
    """

    heatmap = np.asarray(heatmap, dtype=np.float32).squeeze()
    depth_m = np.asarray(depth_m, dtype=np.float32).squeeze()
    object_mask = np.asarray(object_mask).squeeze().astype(bool)
    intrinsic_cv = np.asarray(intrinsic_cv, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be [H,W], got {heatmap.shape}")
    if depth_m.shape != heatmap.shape or object_mask.shape != heatmap.shape:
        raise ValueError("heatmap, depth_m, and object_mask must be spatially aligned")
    if intrinsic_cv.shape != (3, 3):
        raise ValueError(f"intrinsic_cv must be [3,3], got {intrinsic_cv.shape}")
    if not np.all(np.isfinite(heatmap)):
        raise ValueError("heatmap contains non-finite values")
    if heat_threshold < 0 or heat_threshold > 1:
        raise ValueError("heat_threshold must lie in [0,1]")

    valid = (
        object_mask
        & np.isfinite(depth_m)
        & (depth_m >= minimum_depth_m)
        & (depth_m <= maximum_depth_m)
    )
    rows_v, columns_u = np.nonzero(valid)
    if not len(columns_u):
        raise ValueError("No valid object depth pixel is available for lifting")
    depth = depth_m[rows_v, columns_u]
    fx, fy = float(intrinsic_cv[0, 0]), float(intrinsic_cv[1, 1])
    cx, cy = float(intrinsic_cv[0, 2]), float(intrinsic_cv[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be positive")
    points_camera = np.stack(
        (
            (columns_u.astype(np.float32) - cx) * depth / fx,
            (rows_v.astype(np.float32) - cy) * depth / fy,
            depth,
        ),
        axis=-1,
    ).astype(np.float32)
    raw_heat = np.clip(heatmap[rows_v, columns_u], 0.0, 1.0).astype(np.float32)
    heat = np.where(raw_heat >= heat_threshold, raw_heat, 0.0).astype(np.float32)
    if not np.any(heat > 0):
        raise ValueError(
            f"No visible object pixel exceeds heat_threshold={heat_threshold:.3f}"
        )
    return LiftedImageHeat(
        pixels_uv=np.stack((columns_u, rows_v), axis=-1).astype(np.int32),
        points_camera=points_camera,
        heat=heat,
        raw_heat=raw_heat,
    )
