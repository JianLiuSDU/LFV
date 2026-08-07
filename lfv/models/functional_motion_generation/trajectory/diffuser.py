"""Goal-conditioned 64-step trajectory diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lfv.diffusion import Pose9DNormalizer, make_ddim_scheduler, make_ddpm_scheduler
from lfv.geometry import (
    identity_pose9d,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    so3_geodesic_distance,
)

from .decoder import TrajectoryDecoder


class TrajectoryDiffuser(nn.Module):
    def __init__(
        self,
        decoder: TrajectoryDecoder,
        *,
        num_train_timesteps: int = 100,
        inference_steps: int = 20,
        diffusion_weight: float = 1.0,
        translation_weight: float = 1.0,
        rotation_weight: float = 0.5,
        velocity_weight: float = 0.2,
        endpoint_weight: float = 1.0,
        start_reconstruction_weight: float = 1.0,
        start_boundary_weight: float = 0.0,
        acceleration_weight: float = 0.0,
        goal_perturb_std: float = 0.03,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.num_train_timesteps = int(num_train_timesteps)
        self.inference_steps = int(inference_steps)
        self.weights = (
            float(diffusion_weight),
            float(translation_weight),
            float(rotation_weight),
            float(velocity_weight),
            float(endpoint_weight),
            float(start_boundary_weight),
            float(acceleration_weight),
        )
        self.start_reconstruction_weight = float(start_reconstruction_weight)
        self.goal_perturb_std = float(goal_perturb_std)
        self.train_scheduler = make_ddpm_scheduler(
            num_train_timesteps=self.num_train_timesteps
        )

    def _training_goal(
        self,
        normalized_goal: torch.Tensor,
    ) -> torch.Tensor:
        perturb = torch.randn_like(normalized_goal) * self.goal_perturb_std
        keep_clean = (
            torch.rand(normalized_goal.shape[0], 1, device=normalized_goal.device)
            < 0.34
        )
        return torch.where(keep_clean, normalized_goal, normalized_goal + perturb)

    def compute_loss(
        self,
        context: torch.Tensor,
        goal_pose9d: torch.Tensor,
        trajectory_pose9d: torch.Tensor,
        normalizer: Pose9DNormalizer,
    ) -> dict[str, torch.Tensor]:
        clean_full = normalizer.normalize(trajectory_pose9d)
        clean = clean_full[:, 1:]
        normalized_start = clean_full[:, 0]
        normalized_goal = normalizer.normalize(goal_pose9d)
        goal_condition = self._training_goal(normalized_goal)
        noise = torch.randn_like(clean)
        timestep = torch.randint(
            0,
            self.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        noisy = self.train_scheduler.add_noise(clean, noise, timestep)
        prediction = self.decoder(
            noisy,
            timestep,
            context,
            goal_condition,
            normalized_start=normalized_start,
        )
        endpoint_weights = torch.ones_like(clean)
        endpoint_weights[:, 0] = self.start_reconstruction_weight
        endpoint_weights[:, -1] = 2.0
        diffusion = ((prediction - clean).square() * endpoint_weights).mean()
        physical = normalizer.denormalize(prediction)
        target = trajectory_pose9d[:, 1:]
        translation = F.smooth_l1_loss(physical[..., :3], target[..., :3])
        rotation = so3_geodesic_distance(
            rotation_6d_to_matrix(physical[..., 3:9]),
            rotation_6d_to_matrix(target[..., 3:9]),
        ).mean()
        predicted_full_translation = torch.cat(
            (
                torch.zeros_like(physical[:, :1, :3]),
                physical[..., :3],
            ),
            dim=1,
        )
        velocity = F.l1_loss(
            predicted_full_translation[:, 1:] - predicted_full_translation[:, :-1],
            trajectory_pose9d[:, 1:, :3] - trajectory_pose9d[:, :-1, :3],
        )
        predicted_acceleration = (
            predicted_full_translation[:, 2:]
            - 2.0 * predicted_full_translation[:, 1:-1]
            + predicted_full_translation[:, :-2]
        )
        target_acceleration = (
            trajectory_pose9d[:, 2:, :3]
            - 2.0 * trajectory_pose9d[:, 1:-1, :3]
            + trajectory_pose9d[:, :-2, :3]
        )
        acceleration = F.l1_loss(predicted_acceleration, target_acceleration)
        start_normalized = F.mse_loss(prediction[:, 0], clean[:, 0])
        start_translation = F.l1_loss(physical[:, 0, :3], target[:, 0, :3])
        start_rotation = so3_geodesic_distance(
            rotation_6d_to_matrix(physical[:, 0, 3:9]),
            rotation_6d_to_matrix(target[:, 0, 3:9]),
        ).mean()
        start_boundary = start_normalized + start_translation + 0.5 * start_rotation
        endpoint_translation = F.l1_loss(
            physical[:, -1, :3], goal_pose9d[:, :3]
        )
        endpoint_rotation = so3_geodesic_distance(
            rotation_6d_to_matrix(physical[:, -1, 3:9]),
            rotation_6d_to_matrix(goal_pose9d[:, 3:9]),
        ).mean()
        endpoint = endpoint_translation + 0.5 * endpoint_rotation
        wd, wt, wr, wv, we, ws, wa = self.weights
        total = (
            wd * diffusion
            + wt * translation
            + wr * rotation
            + wv * velocity
            + we * endpoint
            + ws * start_boundary
            + wa * acceleration
        )
        return {
            "trajectory_total": total,
            "trajectory_diffusion": diffusion,
            "trajectory_translation": translation,
            "trajectory_rotation": rotation,
            "trajectory_velocity": velocity,
            "trajectory_acceleration": acceleration,
            "trajectory_start_boundary": start_boundary,
            "trajectory_start_translation": start_translation,
            "trajectory_start_rotation": start_rotation,
            "trajectory_endpoint": endpoint,
        }

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        goals: torch.Tensor,
        normalizer: Pose9DNormalizer,
        *,
        num_samples_per_goal: int = 2,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        batch, num_goals, _ = goals.shape
        repeated_context = context[:, None, None].expand(
            batch,
            num_goals,
            num_samples_per_goal,
            context.shape[1],
            context.shape[2],
        ).reshape(-1, context.shape[1], context.shape[2])
        repeated_goals = goals[:, :, None].expand(
            batch, num_goals, num_samples_per_goal, 9
        ).reshape(-1, 9)
        normalized_goals = normalizer.normalize(repeated_goals)
        physical_start = identity_pose9d(
            repeated_goals.shape[0],
            dtype=context.dtype,
            device=context.device,
        )
        normalized_start = normalizer.normalize(physical_start)
        state = torch.randn(
            repeated_goals.shape[0],
            63,
            9,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        scheduler = make_ddim_scheduler(
            num_train_timesteps=self.num_train_timesteps
        )
        scheduler.set_timesteps(
            int(inference_steps or self.inference_steps), device=context.device
        )
        for timestep in scheduler.timesteps:
            timestep_batch = torch.full(
                (state.shape[0],),
                int(timestep),
                device=state.device,
                dtype=torch.long,
            )
            clean_prediction = self.decoder(
                state,
                timestep_batch,
                repeated_context,
                normalized_goals,
                normalized_start=normalized_start,
            )
            state = scheduler.step(
                clean_prediction, timestep, state, eta=0.0, generator=generator
            ).prev_sample
        physical = normalizer.denormalize(state)
        physical = torch.cat(
            (
                physical[..., :3],
                matrix_to_rotation_6d(
                    rotation_6d_to_matrix(physical[..., 3:9])
                ),
            ),
            dim=-1,
        )
        start = identity_pose9d(
            physical.shape[0], 1, dtype=physical.dtype, device=physical.device
        )
        trajectory = torch.cat((start, physical), dim=1)
        return trajectory.reshape(
            batch, num_goals, num_samples_per_goal, 64, 9
        )
