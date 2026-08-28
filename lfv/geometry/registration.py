from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(0)
    target_center = target.mean(0)
    h = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = target_center - r @ source_center
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = r.astype(np.float32)
    transform[:3, 3] = t.astype(np.float32)
    return transform


def rigid_icp_to_visible(
    canonical_points: np.ndarray,
    visible_points_camera: np.ndarray,
    *,
    iterations: int = 20,
    max_correspondence_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Register canonical SAM3D vertices to visible RGB-D points.

    This is a small, deterministic point-to-point ICP adapter.  It is not a
    substitute for a learned pose estimator; the returned RMS and transform
    are saved so callers can reject poor completion alignments.
    """
    source = np.asarray(canonical_points, dtype=np.float32)
    target = np.asarray(visible_points_camera, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("canonical_points must be [N,3] with N>=3")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) < 3:
        raise ValueError("visible_points_camera must be [M,3] with M>=3")
    scale_s = np.linalg.norm(np.percentile(source, 95, axis=0) - np.percentile(source, 5, axis=0))
    scale_t = np.linalg.norm(np.percentile(target, 95, axis=0) - np.percentile(target, 5, axis=0))
    if scale_s <= 1e-8 or scale_t <= 1e-8:
        raise ValueError("Degenerate completion or visible cloud")
    transformed = (source - source.mean(0)) * (scale_t / scale_s) + target.mean(0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] *= np.float32(scale_t / scale_s)
    transform[:3, 3] = (target.mean(0) - (scale_t / scale_s) * source.mean(0)).astype(np.float32)
    tree = cKDTree(target)
    for _ in range(max(1, int(iterations))):
        distances, indices = tree.query(transformed)
        keep = np.ones(len(source), dtype=bool) if max_correspondence_m is None else distances <= max_correspondence_m
        if keep.sum() < 3:
            break
        delta = _kabsch(transformed[keep], target[indices[keep]])
        transformed = (delta[:3, :3] @ transformed.T).T + delta[:3, 3]
        transform = delta @ transform
    distances, _ = tree.query(transformed)
    rms = float(np.sqrt(np.mean(distances**2)))
    return transform.astype(np.float32), transformed.astype(np.float32), rms
