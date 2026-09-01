"""Checkpoint construction helpers used by evaluation and simulation."""

from __future__ import annotations

from pathlib import Path

import torch

from lfv.diffusion import Pose9DNormalizer

from .registry import build_model


def model_kwargs(config: dict, dino_dim: int) -> dict:
    model = config["model"]
    return {
        "dino_dim": int(dino_dim),
        "hidden_dim": int(model.get("hidden_dim", 128)),
        "encoder_heads": int(model.get("encoder_heads", 4)),
        "motion_field_mode": str(model.get("motion_field_mode", "none")),
        "motion_field_temperature": float(
            model.get("motion_field_temperature", 1.0)
        ),
        "motion_field_power": float(model.get("motion_field_power", 1.0)),
        "motion_field_pair_weight": float(
            model.get("motion_field_pair_weight", 0.25)
        ),
        "motion_field_fusion_mode": str(
            model.get("motion_field_fusion_mode", "fixed")
        ),
        "motion_field_bottleneck": bool(
            model.get("motion_field_bottleneck", False)
        ),
        "motion_field_causal_weight": float(
            model.get("motion_field_causal_weight", 0.0)
        ),
        "motion_field_causal_margin": float(
            model.get("motion_field_causal_margin", 0.0)
        ),
        "motion_field_drop_top_weight": float(
            model.get("motion_field_drop_top_weight", 0.0)
        ),
        "motion_field_consistency_weight": float(
            model.get("motion_field_consistency_weight", 0.0)
        ),
        "motion_field_consistency_temperature": float(
            model.get("motion_field_consistency_temperature", 0.1)
        ),
        "motion_field_consistency_max_points": int(
            model.get("motion_field_consistency_max_points", 64)
        ),
        "goal_layers": int(model.get("goal_layers", 4)),
        "trajectory_layers": int(model.get("trajectory_layers", 6)),
        "decoder_heads": int(model.get("decoder_heads", 4)),
        "dropout": float(model.get("dropout", 0.1)),
        "num_train_timesteps": int(model.get("num_train_timesteps", 100)),
        "goal_inference_steps": int(model.get("goal_inference_steps", 20)),
        "trajectory_inference_steps": int(
            model.get("trajectory_inference_steps", 20)
        ),
        "trajectory_hard_start_token": bool(
            model.get("trajectory_hard_start_token", False)
        ),
        # Checkpoints written before the position-encoding fix do not carry a
        # version field. Preserve their original [0, 1] behavior rather than
        # silently evaluating learned weights with a new temporal basis.
        "trajectory_position_encoding": str(
            model.get(
                "trajectory_position_encoding",
                "legacy_normalized_sinusoidal",
            )
        ),
        "trajectory_goal_context_layers": int(
            model.get("trajectory_goal_context_layers", 0)
        ),
        "trajectory_goal_context_residual_gating": bool(
            model.get("trajectory_goal_context_residual_gating", False)
        ),
        "trajectory_num_phase_tokens": int(
            model.get("trajectory_num_phase_tokens", 0)
        ),
        "trajectory_temporal_attention_mode": str(
            model.get("trajectory_temporal_attention_mode", "full")
        ),
        "trajectory_temporal_local_window": int(
            model.get("trajectory_temporal_local_window", 7)
        ),
        "trajectory_phase_residual_gating": bool(
            model.get("trajectory_phase_residual_gating", False)
        ),
        "trajectory_residual_gating": bool(
            model.get("trajectory_residual_gating", False)
        ),
        "trajectory_residual_gate_init": float(
            model.get("trajectory_residual_gate_init", 0.1)
        ),
        "trajectory_phase_attention_sigma": float(
            model.get("trajectory_phase_attention_sigma", 0.22)
        ),
        "trajectory_velocity_weight": float(
            model.get("trajectory_velocity_weight", 0.2)
        ),
        "trajectory_endpoint_weight": float(
            model.get("trajectory_endpoint_weight", 1.0)
        ),
        "trajectory_start_reconstruction_weight": float(
            model.get("trajectory_start_reconstruction_weight", 1.0)
        ),
        "trajectory_start_boundary_weight": float(
            model.get("trajectory_start_boundary_weight", 0.0)
        ),
        "trajectory_acceleration_weight": float(
            model.get("trajectory_acceleration_weight", 0.0)
        ),
    }


def load_stage2_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
):
    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    config = payload["config"]
    dino_dim = int(config["data"]["dino_dim"])
    model = build_model(
        config["model"]["name"], **model_kwargs(config, dino_dim)
    )
    model.load_state_dict(payload["model"])
    if use_ema:
        state = dict(model.named_parameters())
        with torch.no_grad():
            for key, value in payload["ema"]["shadow"].items():
                # Old checkpoints may contain floating-point buffers. EMA is
                # only meaningful for learned parameters; dataset statistics
                # must remain the exact values fitted on the training split.
                if key in state and not key.startswith("normalizer."):
                    state[key].copy_(value)
    model.normalizer = Pose9DNormalizer.from_dict(payload["normalizer"])
    model.eval().to(device)
    return model, config, payload
