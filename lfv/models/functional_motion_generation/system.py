"""Hierarchical three-token Stage 2 model."""

from __future__ import annotations

import torch
from torch import nn

from lfv.diffusion import Pose9DNormalizer

from .encoders import BidirectionalSceneEncoder
from .goal import GoalPoseDecoder, GoalPoseDiffuser
from .interfaces import ContextEncoding, Stage2Samples
from .registry import register_model
from .trajectory import TrajectoryDecoder, TrajectoryDiffuser


@register_model("three_token_hierarchical_diffusion")
class ThreeTokenHierarchicalDiffusion(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        encoder_heads: int = 4,
        motion_field_mode: str = "none",
        motion_field_temperature: float = 1.0,
        goal_layers: int = 4,
        trajectory_layers: int = 6,
        decoder_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        goal_inference_steps: int = 20,
        trajectory_inference_steps: int = 20,
        trajectory_hard_start_token: bool = False,
        trajectory_position_encoding: str = "discrete_sinusoidal",
        trajectory_goal_context_layers: int = 0,
        trajectory_goal_context_residual_gating: bool = False,
        trajectory_num_phase_tokens: int = 0,
        trajectory_temporal_attention_mode: str = "full",
        trajectory_temporal_local_window: int = 7,
        trajectory_phase_residual_gating: bool = False,
        trajectory_residual_gating: bool = False,
        trajectory_residual_gate_init: float = 0.1,
        trajectory_phase_attention_sigma: float = 0.22,
        trajectory_velocity_weight: float = 0.2,
        trajectory_endpoint_weight: float = 1.0,
        trajectory_start_reconstruction_weight: float = 1.0,
        trajectory_start_boundary_weight: float = 0.0,
        trajectory_acceleration_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = BidirectionalSceneEncoder(
            dino_dim=dino_dim,
            hidden_dim=hidden_dim,
            dino_projected_dim=hidden_dim // 2,
            xyz_projected_dim=hidden_dim - hidden_dim // 2,
            num_heads=encoder_heads,
            dropout=dropout,
            motion_field_mode=motion_field_mode,
            motion_field_temperature=motion_field_temperature,
        )
        self.goal_diffuser = GoalPoseDiffuser(
            GoalPoseDecoder(
                hidden_dim=hidden_dim,
                num_layers=goal_layers,
                num_heads=decoder_heads,
                dropout=dropout,
            ),
            num_train_timesteps=num_train_timesteps,
            inference_steps=goal_inference_steps,
        )
        self.trajectory_diffuser = TrajectoryDiffuser(
            TrajectoryDecoder(
                hidden_dim=hidden_dim,
                num_layers=trajectory_layers,
                num_heads=decoder_heads,
                dropout=dropout,
                use_hard_start_token=trajectory_hard_start_token,
                position_encoding=trajectory_position_encoding,
                goal_context_layers=trajectory_goal_context_layers,
                goal_context_residual_gating=trajectory_goal_context_residual_gating,
                num_phase_tokens=trajectory_num_phase_tokens,
                temporal_attention_mode=trajectory_temporal_attention_mode,
                temporal_local_window=trajectory_temporal_local_window,
                phase_residual_gating=trajectory_phase_residual_gating,
                residual_gating=trajectory_residual_gating,
                residual_gate_init=trajectory_residual_gate_init,
                phase_attention_sigma=trajectory_phase_attention_sigma,
            ),
            num_train_timesteps=num_train_timesteps,
            inference_steps=trajectory_inference_steps,
            velocity_weight=trajectory_velocity_weight,
            endpoint_weight=trajectory_endpoint_weight,
            start_reconstruction_weight=trajectory_start_reconstruction_weight,
            start_boundary_weight=trajectory_start_boundary_weight,
            acceleration_weight=trajectory_acceleration_weight,
        )
        self.normalizer = Pose9DNormalizer()

    def encode(self, batch: dict, *, return_debug: bool = False) -> ContextEncoding:
        return self.encoder(
            batch["manipulated_points"],
            batch["manipulated_dino"],
            batch["reference_points"],
            batch["reference_dino"],
            return_debug=return_debug,
        )

    def compute_loss(self, batch: dict, stage: str = "joint") -> dict[str, torch.Tensor]:
        encoding = self.encode(batch)
        context = encoding.tokens
        losses: dict[str, torch.Tensor] = {}
        if stage in ("goal", "joint"):
            losses.update(
                self.goal_diffuser.compute_loss(
                    context, batch["goal_pose9d"], self.normalizer
                )
            )
        if stage in ("trajectory", "joint"):
            losses.update(
                self.trajectory_diffuser.compute_loss(
                    context,
                    batch["goal_pose9d"],
                    batch["trajectory_pose9d"],
                    self.normalizer,
                )
            )
        if stage == "goal":
            losses["total"] = losses["goal_total"]
        elif stage == "trajectory":
            losses["total"] = losses["trajectory_total"]
        elif stage == "joint":
            losses["total"] = losses["goal_total"] + losses["trajectory_total"]
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        fields = [
            field
            for field in (
                encoding.manipulated_motion_field,
                encoding.reference_motion_field,
            )
            if field is not None
        ]
        if fields:
            normalized_entropies = []
            peaks = []
            for field in fields:
                entropy = -(field * field.clamp_min(1e-12).log()).sum(dim=1)
                entropy = entropy / torch.log(
                    torch.as_tensor(
                        field.shape[1], device=field.device, dtype=field.dtype
                    )
                )
                normalized_entropies.append(entropy.mean())
                peaks.append(field.max(dim=1).values.mean())
            losses["motion_field_entropy"] = torch.stack(
                normalized_entropies
            ).mean().detach()
            losses["motion_field_peak"] = torch.stack(peaks).mean().detach()
        return losses

    @torch.no_grad()
    def sample(
        self,
        batch: dict,
        *,
        num_goal_samples: int = 8,
        num_trajectory_samples: int = 2,
        generator: torch.Generator | None = None,
        return_debug: bool = False,
        goal_inference_steps: int | None = None,
        trajectory_inference_steps: int | None = None,
    ) -> tuple[Stage2Samples, ContextEncoding]:
        encoding = self.encode(batch, return_debug=return_debug)
        goals = self.goal_diffuser.sample(
            encoding.tokens,
            self.normalizer,
            num_samples=num_goal_samples,
            generator=generator,
            inference_steps=goal_inference_steps,
        )
        trajectories = self.trajectory_diffuser.sample(
            encoding.tokens,
            goals,
            self.normalizer,
            num_samples_per_goal=num_trajectory_samples,
            generator=generator,
            inference_steps=trajectory_inference_steps,
        )
        goal_ids = torch.arange(
            num_goal_samples, device=goals.device, dtype=torch.long
        )[None, :, None].expand(
            goals.shape[0], num_goal_samples, num_trajectory_samples
        )
        return Stage2Samples(goals, trajectories, goal_ids), encoding
