"""Pose9D and coordinate-frame helpers for Stage 2."""

from __future__ import annotations

import numpy as np
import torch

from .rotation6d import matrix_to_rotation_6d, rotation_6d_to_matrix


def matrix_to_pose9d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [...,4,4], got {tuple(matrix.shape)}")
    return torch.cat(
        (matrix[..., :3, 3], matrix_to_rotation_6d(matrix[..., :3, :3])),
        dim=-1,
    )


def pose9d_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 9:
        raise ValueError(f"Expected [...,9], got {tuple(pose.shape)}")
    output = torch.zeros(
        *pose.shape[:-1], 4, 4, dtype=pose.dtype, device=pose.device
    )
    output[..., :3, :3] = rotation_6d_to_matrix(pose[..., 3:9])
    output[..., :3, 3] = pose[..., :3]
    output[..., 3, 3] = 1.0
    return output


def matrix_to_pose9d_np(matrix: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(matrix), dtype=torch.float32)
    return matrix_to_pose9d(tensor).cpu().numpy().astype(np.float32)


def pose9d_to_matrix_np(pose: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(pose), dtype=torch.float32)
    return pose9d_to_matrix(tensor).cpu().numpy().astype(np.float32)


def camera_delta_to_local(
    camera_delta: np.ndarray,
    scene_origin: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    """Change a camera-frame rigid delta into the shared local frame."""

    transform = np.asarray(camera_delta, dtype=np.float32).copy()
    origin = np.asarray(scene_origin, dtype=np.float32).reshape(3)
    transform[:3, 3] = (
        transform[:3, :3] @ origin + transform[:3, 3] - origin
    ) / float(scene_scale)
    return transform


def local_delta_to_camera(
    local_delta: np.ndarray,
    scene_origin: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    """Invert camera_delta_to_local."""

    transform = np.asarray(local_delta, dtype=np.float32).copy()
    origin = np.asarray(scene_origin, dtype=np.float32).reshape(3)
    transform[:3, 3] = (
        transform[:3, 3] * float(scene_scale)
        - transform[:3, :3] @ origin
        + origin
    )
    return transform


def identity_pose9d(
    *batch_shape: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    pose = torch.zeros(*batch_shape, 9, dtype=dtype, device=device)
    pose[..., 3] = 1.0
    pose[..., 7] = 1.0
    return pose
