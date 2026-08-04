"""Data and frame adapters for the two-stage pouring motion models.

The checkpoints live in the historical ``object_centric_diffusion`` project,
but their input/output contract is kept here so that LFV owns the executable
pipeline. Model-specific loading stays in the command-line adapter.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def unproject_pixels(
    pixels_uv: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_cv: np.ndarray,
) -> np.ndarray:
    """Unproject ``[u,v]`` pixels into the OpenCV camera frame."""

    pixels = np.asarray(pixels_uv, dtype=np.int64).reshape(-1, 2)
    depth = np.asarray(depth_m, dtype=np.float32)
    intrinsic = np.asarray(intrinsic_cv, dtype=np.float32)
    u = pixels[:, 0]
    v = pixels[:, 1]
    z = depth[v, u]
    x = (u.astype(np.float32) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v.astype(np.float32) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def _spatially_distributed_indices(
    pixels_uv: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministic score-aware farthest sampling in image space."""

    pixels = np.asarray(pixels_uv, dtype=np.float32).reshape(-1, 2)
    score = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(pixels) == 0:
        raise ValueError("Cannot sample from an empty pixel set")
    score = score - float(score.min())
    score /= max(float(score.max()), 1e-8)
    selected = [int(np.argmax(score))]
    min_distance_sq = np.full(len(pixels), np.inf, dtype=np.float32)
    unique_count = min(int(count), len(pixels))
    for _ in range(1, unique_count):
        delta = pixels - pixels[selected[-1]][None]
        min_distance_sq = np.minimum(
            min_distance_sq,
            np.sum(delta * delta, axis=-1),
        )
        utility = min_distance_sq * (0.25 + 0.75 * score)
        utility[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(utility)))
    if len(selected) < count:
        repeats = np.resize(np.asarray(selected, dtype=np.int64), count - len(selected))
        selected.extend(repeats.tolist())
    return np.asarray(selected, dtype=np.int64)


def sample_heat_point_cloud(
    heatmap: np.ndarray,
    object_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_cv: np.ndarray,
    count: int,
    *,
    heat_quantile: float = 0.60,
    minimum_heat: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a spatially spread point cloud from a continuous contact heatmap."""

    heat = np.asarray(heatmap, dtype=np.float32)
    mask = np.asarray(object_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > 1e-4)
    positive = heat[valid]
    positive = positive[positive >= minimum_heat]
    if len(positive) == 0:
        raise ValueError("No valid depth pixel has contact heat above minimum_heat")
    threshold = max(float(minimum_heat), float(np.quantile(positive, heat_quantile)))
    candidate = valid & (heat >= threshold)
    v, u = np.nonzero(candidate)
    pixels = np.stack((u, v), axis=-1)
    scores = heat[v, u]
    chosen = _spatially_distributed_indices(pixels, scores, count)
    sampled_pixels = pixels[chosen].astype(np.int32)
    sampled_scores = scores[chosen].astype(np.float32)
    points = unproject_pixels(sampled_pixels, depth, intrinsic_cv)
    return points, sampled_pixels, sampled_scores


def sample_mask_point_cloud(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_cv: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample a spatially distributed target point cloud."""

    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > 1e-4)
    v, u = np.nonzero(valid)
    pixels = np.stack((u, v), axis=-1)
    chosen = _spatially_distributed_indices(
        pixels,
        np.ones(len(pixels), dtype=np.float32),
        count,
    )
    sampled_pixels = pixels[chosen].astype(np.int32)
    return unproject_pixels(sampled_pixels, depth, intrinsic_cv), sampled_pixels


def pose7d_xyzw_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix().astype(np.float32)
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose7d_xyzw(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return np.concatenate(
        (
            transform[:3, 3],
            Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32),
        )
    ).astype(np.float32)


def local_pose7d_to_camera_delta(
    local_pose_xyzw: np.ndarray,
    manipulated_centroid_camera: np.ndarray,
) -> np.ndarray:
    """Convert a centroid-local model pose into a camera-frame rigid delta."""

    local = pose7d_xyzw_to_matrix(local_pose_xyzw)
    centroid = np.asarray(manipulated_centroid_camera, dtype=np.float32).reshape(3)
    delta = local.copy()
    delta[:3, 3] = local[:3, 3] + centroid - local[:3, :3] @ centroid
    return delta.astype(np.float32)


def camera_delta_to_world_delta(
    camera_delta: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    world_to_camera = np.asarray(world_to_camera, dtype=np.float32).reshape(4, 4)
    return (
        np.linalg.inv(world_to_camera)
        @ np.asarray(camera_delta, dtype=np.float32).reshape(4, 4)
        @ world_to_camera
    ).astype(np.float32)


def local_trajectory_to_world_object_poses(
    local_poses_xyzw: np.ndarray,
    manipulated_centroid_camera: np.ndarray,
    world_to_camera: np.ndarray,
    object_to_world_initial: np.ndarray,
) -> np.ndarray:
    """Map model-local relative poses to absolute object poses in ManiSkill."""

    output = []
    object_initial = np.asarray(object_to_world_initial, dtype=np.float32).reshape(4, 4)
    for pose in np.asarray(local_poses_xyzw, dtype=np.float32):
        camera_delta = local_pose7d_to_camera_delta(pose, manipulated_centroid_camera)
        world_delta = camera_delta_to_world_delta(camera_delta, world_to_camera)
        output.append(world_delta @ object_initial)
    return np.stack(output).astype(np.float32)


def localize_clouds(
    manipulated_camera: np.ndarray,
    target_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the training convention: both clouds subtract the cup centroid."""

    manipulated = np.asarray(manipulated_camera, dtype=np.float32)
    target = np.asarray(target_camera, dtype=np.float32)
    centroid = manipulated.mean(axis=0).astype(np.float32)
    return manipulated - centroid, target - centroid, centroid
