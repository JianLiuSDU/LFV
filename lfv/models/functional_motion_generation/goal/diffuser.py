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
        candidate_scoring: bool = False,
        score_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.num_train_timesteps = int(num_train_timesteps)
        self.inference_steps = int(inference_steps)
        self.diffusion_weight = float(diffusion_weight)
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.candidate_scoring = bool(candidate_scoring)
        self.score_weight = float(score_weight)
        self.candidate_scorer = None
        # Constructed lazily by ``set_context_dim`` from the parent model.  A
        # scorer is optional so legacy checkpoints keep their exact parameter
        # structure and sampling behavior.
        self.train_scheduler = make_ddpm_scheduler(
            num_train_timesteps=self.num_train_timesteps
        )

    def set_context_dim(self, hidden_dim: int) -> None:
        """Create the tiny candidate scorer once the scene width is known."""

        if not self.candidate_scoring or self.candidate_scorer is not None:
            return
        self.candidate_scorer = nn.Sequential(
            nn.LayerNorm(9 + hidden_dim * 2),
            nn.Linear(9 + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _score_inputs(
        self,
        normalized_goals: torch.Tensor,
        context: torch.Tensor,
        goal_relation_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.candidate_scorer is None:
            raise RuntimeError("candidate scorer is not enabled")
        context_summary = context.mean(dim=1)
        if goal_relation_tokens is None:
            relation_summary = torch.zeros_like(context_summary)
        else:
            relation_summary = goal_relation_tokens.mean(dim=1)
        return torch.cat((normalized_goals, context_summary, relation_summary), dim=-1)

    def compute_loss(
        self,
        context: torch.Tensor,
        goal_pose9d: torch.Tensor,
        normalizer: Pose9DNormalizer,
        goal_relation_tokens: torch.Tensor | None = None,
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
        prediction = self.decoder(
            noisy, timestep, context, goal_relation_tokens=goal_relation_tokens
        )
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
        output = {
            "goal_total": total,
            "goal_diffusion": diffusion,
            "goal_translation": translation,
            "goal_rotation": rotation,
        }
        if self.candidate_scoring:
            if self.candidate_scorer is None:
                raise RuntimeError("candidate scorer was not initialized")
            negative = clean + 0.75 * torch.randn_like(clean)
            score_inputs = torch.cat(
                (
                    self._score_inputs(clean, context, goal_relation_tokens),
                    self._score_inputs(negative, context, goal_relation_tokens),
                ),
                dim=0,
            )
            score_targets = torch.cat(
                (
                    torch.ones(clean.shape[0], 1, device=clean.device),
                    torch.zeros(clean.shape[0], 1, device=clean.device),
                ),
                dim=0,
            )
            score_loss = F.binary_cross_entropy_with_logits(
                self.candidate_scorer(score_inputs), score_targets
            )
            output["goal_score"] = score_loss
            output["goal_total"] = total + self.score_weight * score_loss
        return output

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        normalizer: Pose9DNormalizer,
        *,
        num_samples: int = 8,
        generator: torch.Generator | None = None,
        inference_steps: int | None = None,
        goal_relation_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = context.shape[0]
        repeated_context = context.repeat_interleave(num_samples, dim=0)
        repeated_relation = None
        if goal_relation_tokens is not None:
            repeated_relation = goal_relation_tokens.repeat_interleave(num_samples, dim=0)
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
                state,
                timestep_batch,
                repeated_context,
                goal_relation_tokens=repeated_relation,
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

    @torch.no_grad()
    def score_candidates(
        self,
        goals: torch.Tensor,
        context: torch.Tensor,
        normalizer: Pose9DNormalizer,
        goal_relation_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Score sampled physical goals, returning ``[B,K]`` logits."""

        if self.candidate_scorer is None:
            return None
        batch, count, _ = goals.shape
        normalized = normalizer.normalize(goals.reshape(batch * count, 9))
        repeated_context = context[:, None].expand(
            batch, count, context.shape[1], context.shape[2]
        ).reshape(batch * count, context.shape[1], context.shape[2])
        repeated_relation = None
        if goal_relation_tokens is not None:
            repeated_relation = goal_relation_tokens[:, None].expand(
                batch,
                count,
                goal_relation_tokens.shape[1],
                goal_relation_tokens.shape[2],
            ).reshape(
                batch * count,
                goal_relation_tokens.shape[1],
                goal_relation_tokens.shape[2],
            )
        logits = self.candidate_scorer(
            self._score_inputs(normalized, repeated_context, repeated_relation)
        )
        return logits.reshape(batch, count)
