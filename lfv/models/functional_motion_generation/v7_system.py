"""V7 Source-Canonical Functional Alignment diffusion model.

V7 reuses the existing low-dimensional Goal/Trajectory diffusion heads while
replacing only the scene encoder.  The encoder's generator-facing payload is
always the field-gated functional context; selector hidden states are not
available to either decoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lfv.diffusion import Pose9DNormalizer

from .encoders.v7 import V7SceneEncoder, V7SceneEncoding
from .goal import GoalPoseDecoder, GoalPoseDiffuser
from .interfaces import ContextEncoding, Stage2Samples
from .registry import register_model
from .trajectory import TrajectoryDecoder, TrajectoryDiffuser


@register_model("v7_functional_alignment")
class V7FunctionalAlignmentDiffusion(nn.Module):
    """Field-gated V7 model with the same public Stage 2 interface as V2/V6."""

    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 128,
        encoder_heads: int = 4,
        motion_field_mode: str = "local_functional_bottleneck",
        # V7 encoder
        local_dino_proj_dim: int = 64,
        local_object_xyz_dim: int = 32,
        local_relation_xyz_dim: int = 32,
        field_selector_layers: int = 2,
        field_selector_ffn_dim: int = 256,
        field_selector_dropout: float = 0.1,
        field_temperature_start: float = 1.0,
        field_temperature_end: float = 0.4,
        gated_relation_layers: int = 2,
        gated_relation_ffn_dim: int = 256,
        functional_pooling_queries: int = 4,
        field_target_ratio_start: float = 0.5,
        field_target_ratio_end: float = 0.2,
        field_knn: int = 8,
        field_budget_weight: float = 0.02,
        field_smooth_weight: float = 0.01,
        field_consistency_weight: float = 0.02,
        field_consistency_temperature: float = 0.15,
        field_consistency_max_points: int = 64,
        # Existing diffusion heads
        goal_layers: int = 4,
        trajectory_layers: int = 6,
        decoder_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        goal_inference_steps: int = 20,
        trajectory_inference_steps: int = 20,
        trajectory_hard_start_token: bool = True,
        trajectory_position_encoding: str = "discrete_sinusoidal",
        trajectory_goal_context_layers: int = 2,
        trajectory_goal_context_residual_gating: bool = True,
        trajectory_num_phase_tokens: int = 4,
        trajectory_temporal_attention_mode: str = "full",
        trajectory_temporal_local_window: int = 7,
        trajectory_phase_residual_gating: bool = True,
        trajectory_residual_gating: bool = False,
        trajectory_residual_gate_init: float = 0.1,
        trajectory_phase_attention_sigma: float = 0.22,
        trajectory_velocity_weight: float = 0.5,
        trajectory_endpoint_weight: float = 1.0,
        trajectory_start_reconstruction_weight: float = 20.0,
        trajectory_start_boundary_weight: float = 2.0,
        trajectory_acceleration_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if motion_field_mode not in {"local_functional_bottleneck", "v7", "joint"}:
            raise ValueError(
                "V7 motion_field_mode must be local_functional_bottleneck, v7, or joint"
            )
        if not 0.0 <= field_target_ratio_end <= 1.0:
            raise ValueError("field_target_ratio_end must be in [0,1]")
        self.encoder = V7SceneEncoder(
            dino_dim=dino_dim,
            hidden_dim=hidden_dim,
            num_heads=encoder_heads,
            local_dino_proj_dim=local_dino_proj_dim,
            local_object_xyz_dim=local_object_xyz_dim,
            local_relation_xyz_dim=local_relation_xyz_dim,
            selector_layers=field_selector_layers,
            selector_ffn_dim=field_selector_ffn_dim,
            selector_dropout=field_selector_dropout,
            relation_layers=gated_relation_layers,
            relation_ffn_dim=gated_relation_ffn_dim,
            pooling_queries=functional_pooling_queries,
            dropout=dropout,
            field_temperature=field_temperature_start,
        )
        self.field_temperature_start = float(field_temperature_start)
        self.field_temperature_end = float(field_temperature_end)
        self.field_target_ratio_start = float(field_target_ratio_start)
        self.field_target_ratio_end = float(field_target_ratio_end)
        self.field_knn = int(field_knn)
        self.field_budget_weight = float(field_budget_weight)
        self.field_smooth_weight = float(field_smooth_weight)
        self.field_consistency_weight = float(field_consistency_weight)
        self.field_consistency_temperature = float(field_consistency_temperature)
        self.field_consistency_max_points = int(field_consistency_max_points)
        self.training_progress = 0.0
        self.field_target_ratio = self.field_target_ratio_start
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
        self.set_training_progress(0.0)

    def set_training_progress(self, progress: float) -> None:
        """Update the V7 selection curriculum using normalized progress."""

        progress = float(min(max(progress, 0.0), 1.0))
        self.training_progress = progress
        self.field_target_ratio = (
            self.field_target_ratio_start
            + progress * (self.field_target_ratio_end - self.field_target_ratio_start)
        )
        temperature = (
            self.field_temperature_start
            + progress * (self.field_temperature_end - self.field_temperature_start)
        )
        with torch.no_grad():
            self.encoder.selector.temperature.copy_(
                torch.tensor(
                    max(temperature, 1e-3),
                    device=self.encoder.selector.temperature.device,
                    dtype=self.encoder.selector.temperature.dtype,
                ).log()
            )

    @staticmethod
    def _mask(batch: dict, key: str, reference: torch.Tensor) -> torch.Tensor:
        mask = batch.get(key)
        if mask is None:
            return torch.ones(
                reference.shape[:2], device=reference.device, dtype=reference.dtype
            )
        return mask.to(reference).clamp(0.0, 1.0)

    def encode(
        self,
        batch: dict,
        *,
        return_debug: bool = False,
        motion_field_intervention: str | None = None,
        field_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ContextEncoding:
        raw = self.encoder(
            batch["manipulated_points"],
            batch["manipulated_dino"],
            batch["reference_points"],
            batch["reference_dino"],
            manipulated_mask=self._mask(batch, "manipulated_mask", batch["manipulated_points"]),
            reference_mask=self._mask(batch, "reference_mask", batch["reference_points"]),
            scene_scale=batch.get("scene_scale"),
            field_override=field_override,
            field_intervention=motion_field_intervention,
            return_debug=return_debug,
        )
        return raw.context if isinstance(raw, V7SceneEncoding) else raw

    @staticmethod
    def _budget(gate: torch.Tensor, mask: torch.Tensor, target: float) -> torch.Tensor:
        ratio = (gate * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return (ratio - float(target)).square().mean()

    def _smoothness(
        self, points: torch.Tensor, gate: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        n_points = points.shape[1]
        k = min(max(self.field_knn, 1), max(n_points - 1, 1))
        if n_points < 2 or k < 1:
            return gate.new_zeros(())
        distances = torch.cdist(points.float(), points.float())
        pair_valid = mask[:, :, None] * mask[:, None, :]
        distances = distances.masked_fill(pair_valid <= 0.5, float("inf"))
        diagonal = torch.eye(n_points, device=points.device, dtype=torch.bool)[None]
        distances = distances.masked_fill(diagonal, float("inf"))
        indices = distances.topk(k=k, dim=-1, largest=False).indices
        neighbor_gate = gate.gather(1, indices.reshape(points.shape[0], -1)).reshape(
            points.shape[0], n_points, k
        )
        difference = (neighbor_gate - gate[:, :, None]).abs()
        weights = mask[:, :, None]
        return (difference * weights).sum() / weights.sum().clamp_min(1.0) / k

    @staticmethod
    def _symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(1e-8)
        q = q.clamp_min(1e-8)
        return 0.5 * (
            (p * (p.log() - q.log())).sum(-1)
            + (q * (q.log() - p.log())).sum(-1)
        )

    def _field_consistency_loss(
        self, batch: dict, encoding: ContextEncoding
    ) -> torch.Tensor | None:
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
        losses: list[torch.Tensor] = []
        for left, right in pairs:
            role_losses: list[torch.Tensor] = []
            for prefix, field in zip(("manipulated", "reference"), fields):
                assert field is not None
                dino = F.normalize(batch[prefix + "_dino"].float(), dim=-1)
                mask = self._mask(batch, prefix + "_mask", dino)
                f_left = (field[left] * mask[left]).clamp_min(1e-8)
                f_right = (field[right] * mask[right]).clamp_min(1e-8)
                f_left = f_left / f_left.sum().clamp_min(1e-8)
                f_right = f_right / f_right.sum().clamp_min(1e-8)
                affinity = dino[left] @ dino[right].transpose(0, 1)
                temperature = max(self.field_consistency_temperature, 1e-4)
                p_lr = torch.softmax(affinity / temperature, dim=-1)
                p_rl = torch.softmax(affinity.transpose(0, 1) / temperature, dim=-1)
                left_to_right = p_lr.transpose(0, 1) @ f_left
                right_to_left = p_rl.transpose(0, 1) @ f_right
                role_losses.append(
                    0.5
                    * (
                        self._symmetric_kl(left_to_right, f_right)
                        + self._symmetric_kl(right_to_left, f_left)
                    )
                )
            losses.append(torch.stack(role_losses).mean())
        return torch.stack(losses).mean()

    def _field_losses(self, batch: dict, encoding: ContextEncoding) -> dict[str, torch.Tensor]:
        if (
            encoding.manipulated_motion_field is None
            or encoding.reference_motion_field is None
        ):
            return {}
        points_m = batch["manipulated_points"]
        points_r = batch["reference_points"]
        mask_m = self._mask(batch, "manipulated_mask", points_m)
        mask_r = self._mask(batch, "reference_mask", points_r)
        budget_m = self._budget(
            encoding.manipulated_motion_field, mask_m, self.field_target_ratio
        )
        budget_r = self._budget(
            encoding.reference_motion_field, mask_r, self.field_target_ratio
        )
        smooth_m = self._smoothness(
            points_m, encoding.manipulated_motion_field, mask_m
        )
        smooth_r = self._smoothness(
            points_r, encoding.reference_motion_field, mask_r
        )
        losses = {
            "field_budget_manipulated": budget_m,
            "field_budget_reference": budget_r,
            "field_budget": 0.5 * (budget_m + budget_r),
            "field_smoothness_manipulated": smooth_m,
            "field_smoothness_reference": smooth_r,
            "field_smoothness": 0.5 * (smooth_m + smooth_r),
            "field_selected_ratio_manipulated": (
                (encoding.manipulated_motion_field * mask_m).sum(dim=1)
                / mask_m.sum(dim=1).clamp_min(1.0)
            ).mean().detach(),
            "field_selected_ratio_reference": (
                (encoding.reference_motion_field * mask_r).sum(dim=1)
                / mask_r.sum(dim=1).clamp_min(1.0)
            ).mean().detach(),
        }
        consistency = self._field_consistency_loss(batch, encoding)
        if consistency is not None:
            losses["field_consistency"] = consistency
        return losses

    def compute_loss(self, batch: dict, stage: str = "joint") -> dict[str, torch.Tensor]:
        encoding = self.encode(batch)
        losses: dict[str, torch.Tensor] = {}
        if stage in ("goal", "joint"):
            losses.update(
                self.goal_diffuser.compute_loss(
                    encoding.tokens, batch["goal_pose9d"], self.normalizer
                )
            )
        if stage in ("trajectory", "joint"):
            losses.update(
                self.trajectory_diffuser.compute_loss(
                    encoding.tokens,
                    batch["goal_pose9d"],
                    batch["trajectory_pose9d"],
                    self.normalizer,
                )
            )
        if stage == "goal":
            total = losses["goal_total"]
        elif stage == "trajectory":
            total = losses["trajectory_total"]
        elif stage == "joint":
            total = losses["goal_total"] + losses["trajectory_total"]
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        field_losses = self._field_losses(batch, encoding)
        losses.update(field_losses)
        if "field_budget" in field_losses:
            total = total + self.field_budget_weight * field_losses["field_budget"]
            total = total + self.field_smooth_weight * field_losses["field_smoothness"]
            if "field_consistency" in field_losses:
                total = total + self.field_consistency_weight * field_losses["field_consistency"]
        losses["total"] = total
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
        field_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[Stage2Samples, ContextEncoding]:
        encoding = self.encode(
            batch,
            return_debug=return_debug,
            motion_field_intervention=motion_field_intervention,
            field_override=field_override,
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
