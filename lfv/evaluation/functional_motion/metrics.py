"""Physical Goal and trajectory metrics."""

from __future__ import annotations

import torch

from lfv.geometry import rotation_6d_to_matrix, so3_geodesic_distance


def goal_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    target = target[:, None]
    translation = torch.linalg.norm(predicted[..., :3] - target[..., :3], dim=-1)
    rotation = so3_geodesic_distance(
        rotation_6d_to_matrix(predicted[..., 3:9]),
        rotation_6d_to_matrix(target[..., 3:9]),
    )
    combined = translation + 0.05 * rotation
    best = combined.argmin(dim=1)
    rows = torch.arange(predicted.shape[0], device=predicted.device)
    return {
        "goal_top1_translation_m": translation[:, 0].mean(),
        "goal_top1_rotation_deg": torch.rad2deg(rotation[:, 0]).mean(),
        "goal_best_translation_m": translation[rows, best].mean(),
        "goal_best_rotation_deg": torch.rad2deg(rotation[rows, best]).mean(),
    }


def trajectory_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    translation = torch.linalg.norm(predicted[..., :3] - target[..., :3], dim=-1)
    rotation = so3_geodesic_distance(
        rotation_6d_to_matrix(predicted[..., 3:9]),
        rotation_6d_to_matrix(target[..., 3:9]),
    )
    velocity_pred = predicted[:, 1:, :3] - predicted[:, :-1, :3]
    velocity_gt = target[:, 1:, :3] - target[:, :-1, :3]
    return {
        "trajectory_translation_m": translation.mean(),
        "trajectory_rotation_deg": torch.rad2deg(rotation).mean(),
        "trajectory_endpoint_translation_m": translation[:, -1].mean(),
        "trajectory_endpoint_rotation_deg": torch.rad2deg(rotation[:, -1]).mean(),
        "trajectory_velocity_l1_m": (velocity_pred - velocity_gt).abs().mean(),
    }


def trajectory_best_of_k_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Report top-1 and oracle Best-of-K for ``[B,G,K,T,9]`` samples."""

    batch = predicted.shape[0]
    flattened = predicted.reshape(batch, -1, predicted.shape[-2], 9)
    expanded_target = target[:, None]
    translation = torch.linalg.norm(
        flattened[..., :3] - expanded_target[..., :3], dim=-1
    )
    rotation = so3_geodesic_distance(
        rotation_6d_to_matrix(flattened[..., 3:9]),
        rotation_6d_to_matrix(expanded_target[..., 3:9]),
    )
    translation_mean = translation.mean(dim=-1)
    rotation_mean = rotation.mean(dim=-1)
    endpoint_translation = translation[..., -1]
    endpoint_rotation = rotation[..., -1]
    first_step_translation = translation[..., 1]
    first_step_rotation = rotation[..., 1]
    predicted_first_step_magnitude = torch.linalg.norm(
        flattened[..., 1, :3] - flattened[..., 0, :3], dim=-1
    )
    target_first_step_magnitude = torch.linalg.norm(
        target[:, 1, :3] - target[:, 0, :3], dim=-1
    )
    combined = (
        translation_mean
        + 0.05 * rotation_mean
        + endpoint_translation
        + 0.05 * endpoint_rotation
    )
    best = combined.argmin(dim=1)
    rows = torch.arange(batch, device=predicted.device)
    return {
        "trajectory_top1_translation_m": translation_mean[:, 0].mean(),
        "trajectory_top1_rotation_deg": torch.rad2deg(rotation_mean[:, 0]).mean(),
        "trajectory_top1_endpoint_translation_m": endpoint_translation[:, 0].mean(),
        "trajectory_top1_endpoint_rotation_deg": torch.rad2deg(endpoint_rotation[:, 0]).mean(),
        "trajectory_top1_first_step_translation_error_m": first_step_translation[:, 0].mean(),
        "trajectory_top1_first_step_rotation_error_deg": torch.rad2deg(first_step_rotation[:, 0]).mean(),
        "trajectory_top1_predicted_first_step_m": predicted_first_step_magnitude[:, 0].mean(),
        "trajectory_gt_first_step_m": target_first_step_magnitude.mean(),
        "trajectory_best_translation_m": translation_mean[rows, best].mean(),
        "trajectory_best_rotation_deg": torch.rad2deg(rotation_mean[rows, best]).mean(),
        "trajectory_best_endpoint_translation_m": endpoint_translation[rows, best].mean(),
        "trajectory_best_endpoint_rotation_deg": torch.rad2deg(endpoint_rotation[rows, best]).mean(),
        "trajectory_best_first_step_translation_error_m": first_step_translation[rows, best].mean(),
        "trajectory_best_first_step_rotation_error_deg": torch.rad2deg(first_step_rotation[rows, best]).mean(),
        "trajectory_best_predicted_first_step_m": predicted_first_step_magnitude[rows, best].mean(),
    }
