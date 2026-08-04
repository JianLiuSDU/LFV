from __future__ import annotations

import numpy as np


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def depth_to_points_camera(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = normalize_depth(depth)
    height, width = z.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        raise ValueError("Cannot sample an empty object mask")
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int64)


def uniform_grid_sample_mask_pixels(
    mask: np.ndarray,
    num_samples: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    x_min, y_min, x_max, y_max = mask_bbox(mask)
    area = max(1, (x_max - x_min) * (y_max - y_min))
    spacing = max(1, int(np.sqrt(area / max(num_samples * 2, 1))))
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, spacing),
        np.arange(y_min, y_max, spacing),
    )
    pixels = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)
    pixels = pixels[mask[pixels[:, 1], pixels[:, 0]]]
    if not len(pixels):
        raise ValueError("No sampled pixels lie inside the object mask")
    indices = rng.choice(
        len(pixels),
        num_samples,
        replace=len(pixels) < num_samples,
    )
    return pixels[indices].astype(np.int64)


def pixels_to_points_camera(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pixels_uv: np.ndarray,
) -> np.ndarray:
    depth = normalize_depth(depth)
    pixels_uv = np.asarray(pixels_uv, dtype=np.int64)
    u = pixels_uv[:, 0]
    v = pixels_uv[:, 1]
    z = depth[v, u]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack((x, y, z), axis=-1).astype(np.float32)
