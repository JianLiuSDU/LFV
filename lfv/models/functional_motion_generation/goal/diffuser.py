"""DDPM-trained, DDIM-sampled Goal Pose diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lfv.diffusion import Pose9DNormalizer, make_ddim_scheduler, make_ddpm_scheduler
from lfv.geometry import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    so3_geodesic_distance,
)

from .decoder import GoalPoseDecoder


class GoalPoseDiffuser(nn.Module):
    def __init__(
        self,
        decoder: GoalPoseDecoder,
        *,
        num_train_timesteps: int = 100,
        inference_steps: int = 20,
        diffusion_weight: float = 1.0,
        translation_weight: float = 1.0,
        rotation_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.num_train_timesteps = int(num_train_timesteps)
        self.inference_steps = int(inference_steps)
        self.diffusion_weight = float(diffusion_weight)
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.train_scheduler = make_ddpm_scheduler(
            num_train_timesteps=self.num_train_timesteps
        )

    def compute_loss(
        self,
        context: torch.Tensor,
        goal_pose9d: torch.Tensor,
        normalizer: Pose9DNormalizer,
    ) -> dict[str, torch.Tensor]:
        clean = normalizer.normalize(goal_pose9d)
        noise = torch.randn_like(clean)
        timestep = torch.randint(
            0,
            self.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        noisy = self.train_scheduler.add_noise(clean, noise, timestep)
        prediction = self.decoder(noisy, timestep, context)
        diffusion = F.mse_loss(prediction, clean)
        physical = normalizer.denormalize(prediction)
        translation = F.smooth_l1_loss(physical[:, :3], goal_pose9d[:, :3])
        rotation = so3_geodesic_distance(
            rotation_6d_to_matrix(physical[:, 3:9]),
            rotation_6d_to_matrix(goal_pose9d[:, 3:9]),
        ).mean()
        total = (
            self.diffusion_weight * diffusion
            + self.translation_weight * translation
            + self.rotation_weight * rotation
        )
        return {
            "goal_total": total,
            "goal_diffusion": diffusion,
            "goal_translation": translation,
            "goal_rotation": rotation,
        }

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        normalizer: Pose9DNormalizer,
        *,
        num_samples: int = 8,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        batch = context.shape[0]
        repeated_context = context.repeat_interleave(num_samples, dim=0)
        state = torch.randn(
            batch * num_samples,
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
                state, timestep_batch, repeated_context
            )
            state = scheduler.step(
                clean_prediction, timestep, state, eta=0.0, generator=generator
            ).prev_sample
        physical = normalizer.denormalize(state)
        rotation = matrix_to_rotation_6d(
            rotation_6d_to_matrix(physical[:, 3:9])
        )
        physical = torch.cat((physical[:, :3], rotation), dim=-1)
        return physical.reshape(batch, num_samples, 9)
