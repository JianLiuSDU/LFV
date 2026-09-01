"""Hierarchical three-token Stage 2 model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lfv.diffusion import Pose9DNormalizer

from .encoders import BidirectionalSceneEncoder
from .goal import GoalPoseDecoder, GoalPoseDiffuser
from .interfaces import ContextEncoding, Stage2Samples
from .registry import register_model
from .trajectory import TrajectoryDecoder, TrajectoryDiffuser


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@register_model("three_token_hierarchical_diffusion")
class ThreeTokenHierarchicalDiffusion(nn.Module):
    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        encoder_heads: int = 4,
        motion_field_mode: str = "none",
        motion_field_temperature: float = 1.0,
        motion_field_pair_weight: float = 0.25,
        motion_field_fusion_mode: str = "fixed",
        motion_field_bottleneck: bool = False,
        motion_field_causal_weight: float = 0.0,
        motion_field_causal_margin: float = 0.0,
        motion_field_consistency_weight: float = 0.0,
        motion_field_consistency_temperature: float = 0.1,
        motion_field_consistency_max_points: int = 64,
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
            motion_field_pair_weight=motion_field_pair_weight,
            motion_field_fusion_mode=motion_field_fusion_mode,
            motion_field_bottleneck=motion_field_bottleneck,
        )
        self.motion_field_causal_weight = float(motion_field_causal_weight)
        self.motion_field_causal_margin = float(motion_field_causal_margin)
        self.motion_field_consistency_weight = float(motion_field_consistency_weight)
        self.motion_field_consistency_temperature = float(
            motion_field_consistency_temperature
        )
        self.motion_field_consistency_max_points = int(motion_field_consistency_max_points)
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

    def encode(
        self,
        batch: dict,
        *,
        return_debug: bool = False,
        motion_field_intervention: str | None = None,
        motion_field_prior: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        motion_field_prior_weight: float = 0.0,
    ) -> ContextEncoding:
        return self.encoder(
            batch["manipulated_points"],
            batch["manipulated_dino"],
            batch["reference_points"],
            batch["reference_dino"],
            return_debug=return_debug,
            motion_field_intervention=motion_field_intervention,
            motion_field_prior=motion_field_prior,
            motion_field_prior_weight=motion_field_prior_weight,
        )

    def _compute_losses_from_encoding(
        self,
        batch: dict,
        encoding: ContextEncoding,
        stage: str,
    ) -> dict[str, torch.Tensor]:
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

    @staticmethod
    def _symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(1e-8)
        q = q.clamp_min(1e-8)
        return 0.5 * ((p * (p.log() - q.log())).sum(-1) + (q * (q.log() - p.log())).sum(-1))

    def _field_consistency_loss(
        self,
        batch: dict,
        encoding: ContextEncoding,
    ) -> torch.Tensor | None:
        """Compare fields from same-instance demos through soft DINO matching.

        Point order is intentionally ignored: a row-softmax DINO affinity
        transports one demo's field to the other before the symmetric KL.
        Only groups explicitly supplied by the dataset are paired.
        """
        fields = (
            encoding.manipulated_motion_field,
            encoding.reference_motion_field,
        )
        groups = batch.get("field_consistency_group")
        if any(field is None for field in fields) or not groups:
            return None
        pairs: list[tuple[int, int]] = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if str(groups[i]) == str(groups[j]):
                    pairs.append((i, j))
                    if len(pairs) >= 16:
                        break
            if len(pairs) >= 16:
                break
        if not pairs:
            return None
        max_points = min(
            int(self.motion_field_consistency_max_points),
            int(batch["manipulated_dino"].shape[1]),
            int(batch["reference_dino"].shape[1]),
        )
        if max_points < 2:
            return None
        indices = torch.linspace(
            0,
            batch["manipulated_dino"].shape[1] - 1,
            max_points,
            device=batch["manipulated_dino"].device,
        ).long()
        losses: list[torch.Tensor] = []
        for left, right in pairs:
            pair_losses: list[torch.Tensor] = []
            for dino, field in zip(
                (batch["manipulated_dino"], batch["reference_dino"]), fields
            ):
                assert field is not None
                d_left = F.normalize(dino[left, indices].float(), dim=-1)
                d_right = F.normalize(dino[right, indices].float(), dim=-1)
                f_left = field[left, indices].clamp_min(1e-8)
                f_right = field[right, indices].clamp_min(1e-8)
                f_left = f_left / f_left.sum()
                f_right = f_right / f_right.sum()
                affinity = d_left @ d_right.transpose(0, 1)
                temperature = max(float(self.motion_field_consistency_temperature), 1e-4)
                p_left_right = torch.softmax(affinity / temperature, dim=-1)
                p_right_left = torch.softmax(affinity.transpose(0, 1) / temperature, dim=-1)
                transported_left = p_left_right.transpose(0, 1) @ f_left
                transported_right = p_right_left.transpose(0, 1) @ f_right
                pair_losses.append(
                    0.5
                    * (
                        self._symmetric_kl(transported_left, f_right)
                        + self._symmetric_kl(transported_right, f_left)
                    )
                )
            losses.append(torch.stack(pair_losses).mean())
        return torch.stack(losses).mean()

    def compute_loss(self, batch: dict, stage: str = "joint") -> dict[str, torch.Tensor]:
        rng_before = _capture_rng_state()
        encoding = self.encode(batch)
        losses = self._compute_losses_from_encoding(batch, encoding, stage)
        consistency = self._field_consistency_loss(batch, encoding)
        if consistency is not None and self.motion_field_consistency_weight > 0.0:
            losses["motion_field_consistency"] = consistency
            losses["total"] = losses["total"] + self.motion_field_consistency_weight * consistency
        if self.motion_field_causal_weight > 0.0 and encoding.manipulated_motion_field is not None:
            # Re-run the uniform-field intervention with exactly the same
            # dropout/noise/timestep stream, then restore the post-baseline RNG
            # so enabling the probe does not alter the training trajectory.
            baseline_after = _capture_rng_state()
            _restore_rng_state(rng_before)
            uniform_encoding = self.encode(batch, motion_field_intervention="uniform")
            uniform_losses = self._compute_losses_from_encoding(batch, uniform_encoding, stage)
            _restore_rng_state(baseline_after)
            causal = F.relu(
                self.motion_field_causal_margin
                + losses["total"]
                - uniform_losses["total"]
            )
            losses["motion_field_uniform_total"] = uniform_losses["total"].detach()
            losses["motion_field_causal"] = causal
            losses["total"] = losses["total"] + self.motion_field_causal_weight * causal
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
        motion_field_intervention: str | None = None,
        motion_field_prior: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        motion_field_prior_weight: float = 0.0,
    ) -> tuple[Stage2Samples, ContextEncoding]:
        encoding = self.encode(
            batch,
            return_debug=return_debug,
            motion_field_intervention=motion_field_intervention,
            motion_field_prior=motion_field_prior,
            motion_field_prior_weight=motion_field_prior_weight,
        )
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
