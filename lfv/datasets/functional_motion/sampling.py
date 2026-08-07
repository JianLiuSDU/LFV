"""Joint pixel, XYZ and descriptor sampling utilities."""

from __future__ import annotations

import numpy as np


def valid_mask_pixels(
    mask: np.ndarray,
    depth_m: np.ndarray,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = (
        mask
        & np.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    v, u = np.nonzero(valid)
    return np.stack((u, v), axis=-1).astype(np.int32)


def farthest_pixel_sample(pixels_uv: np.ndarray, count: int) -> np.ndarray:
    """Deterministic image-space FPS without replacement."""

    pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
    if len(pixels) < count:
        raise ValueError(f"Need {count} valid unique pixels, found {len(pixels)}")
    center = pixels.mean(axis=0, keepdims=True)
    selected = [int(np.argmin(np.sum((pixels - center) ** 2, axis=-1)))]
    min_distance = np.full(len(pixels), np.inf, dtype=np.float32)
    used = np.zeros(len(pixels), dtype=bool)
    used[selected[0]] = True
    for _ in range(1, int(count)):
        delta = pixels - pixels[selected[-1]][None]
        min_distance = np.minimum(min_distance, np.sum(delta * delta, axis=-1))
        min_distance[used] = -1.0
        index = int(np.argmax(min_distance))
        selected.append(index)
        used[index] = True
    return np.asarray(pixels_uv, dtype=np.int32)[np.asarray(selected, dtype=np.int64)]


def unproject_pixels(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.int64)
    depth = np.asarray(depth_m, dtype=np.float32)
    camera = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    u, v = pixels[:, 0], pixels[:, 1]
    z = depth[v, u]
    x = (u.astype(np.float32) - camera[0, 2]) * z / camera[0, 0]
    y = (v.astype(np.float32) - camera[1, 2]) * z / camera[1, 1]
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def assert_unique_aligned(
    pixels_uv: np.ndarray,
    points: np.ndarray,
    dino: np.ndarray,
    expected: int,
) -> None:
    if pixels_uv.shape != (expected, 2):
        raise ValueError(f"pixels expected {(expected,2)}, got {pixels_uv.shape}")
    if points.shape != (expected, 3) or dino.shape[0] != expected:
        raise ValueError(
            f"unaligned points/DINO: points={points.shape}, dino={dino.shape}"
        )
    if np.unique(pixels_uv, axis=0).shape[0] != expected:
        raise ValueError("Pixel sampling contains repeated indices")
    if np.unique(np.round(points, 6), axis=0).shape[0] < int(expected * 0.99):
        raise ValueError("3D point sampling has too many repeated coordinates")
    if not np.isfinite(points).all() or not np.isfinite(dino).all():
        raise ValueError("Point or DINO data contains NaN/Inf")
