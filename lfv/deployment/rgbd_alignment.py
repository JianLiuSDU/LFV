"""RGB/depth resolution alignment used by camera inference."""

from __future__ import annotations

import cv2
import numpy as np


def align_depth_to_rgb(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_cv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return RGB-aligned depth and correspondingly scaled intrinsics.

    If both images already have the same resolution, this is a validated
    no-op. Without depth intrinsics/extrinsics, resizing is the only safe
    geometric operation available; nearest-neighbor preserves metric depth.
    """

    rgb = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32).squeeze()
    k = np.asarray(intrinsic_cv, dtype=np.float32).copy()
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.ndim != 2:
        raise ValueError(f"Expected RGB [H,W,3] and depth [h,w], got {rgb.shape}, {depth.shape}")
    h, w = rgb.shape[:2]
    dh, dw = depth.shape
    if (dh, dw) == (h, w):
        return depth, k, {"resized": False, "source_shape": [dh, dw], "target_shape": [h, w]}
    sx, sy = float(w) / max(dw, 1), float(h) / max(dh, 1)
    aligned = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    k[0, 0] *= sx; k[0, 2] *= sx
    k[1, 1] *= sy; k[1, 2] *= sy
    return aligned.astype(np.float32), k, {"resized": True, "source_shape": [dh, dw], "target_shape": [h, w], "scale_xy": [sx, sy], "method": "nearest_depth_and_intrinsic_scaling"}
