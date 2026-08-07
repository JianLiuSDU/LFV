"""Continuous 6D rotation representation used by LFV Stage 2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert two rotation columns to an orthonormal rotation matrix."""

    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected [...,6], got {tuple(rotation_6d.shape)}")
    first = rotation_6d[..., 0:3]
    second = rotation_6d[..., 3:6]
    col0 = F.normalize(first, dim=-1, eps=1e-8)
    col1 = F.normalize(
        second - (col0 * second).sum(dim=-1, keepdim=True) * col0,
        dim=-1,
        eps=1e-8,
    )
    col2 = torch.cross(col0, col1, dim=-1)
    return torch.stack((col0, col1, col2), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Store the first two matrix columns as [col0, col1]."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3], got {tuple(matrix.shape)}")
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def project_rotation_6d(rotation_6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(rotation_6d_to_matrix(rotation_6d))


def so3_geodesic_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return SO(3) geodesic distance in radians."""

    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + eps, 1.0 - eps
    )
    return torch.acos(cosine)
