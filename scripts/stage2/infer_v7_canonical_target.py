#!/usr/bin/env python3
"""Run V7 Goal/Trajectory diffusion on a source-canonical target alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lfv.models.functional_motion_generation import load_stage2_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--aligned-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-goals", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device)
    with np.load(args.aligned_target, allow_pickle=False) as data:
        required = (
            "manipulated_points", "manipulated_dino", "manipulated_mask",
            "reference_points", "reference_dino", "reference_mask",
            "manipulated_field_gate", "reference_field_gate",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"aligned target is missing {missing}")
        batch = {
            key: torch.from_numpy(np.asarray(data[key], dtype=np.float32))[None].to(device)
            for key in (
                "manipulated_points", "manipulated_dino", "manipulated_mask",
                "reference_points", "reference_dino", "reference_mask",
            )
        }
        overrides = (
            torch.from_numpy(np.asarray(data["manipulated_field_gate"], dtype=np.float32))[None].to(device),
            torch.from_numpy(np.asarray(data["reference_field_gate"], dtype=np.float32))[None].to(device),
        )
    model, config, _ = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=True
    )
    if config["model"]["name"] != "v7_functional_alignment":
        raise ValueError("--checkpoint must be a V7 functional-alignment checkpoint")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        samples, encoding = model.sample(
            batch,
            num_goal_samples=args.num_goals,
            num_trajectory_samples=args.num_trajectories,
            generator=generator,
            return_debug=True,
            field_override=overrides,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        goals=samples.goals[0].cpu().numpy(),
        trajectories=samples.trajectories[0].cpu().numpy(),
        manipulated_field=encoding.manipulated_motion_field[0].cpu().numpy(),
        reference_field=encoding.reference_motion_field[0].cpu().numpy(),
        goal_ids=samples.goal_ids[0].cpu().numpy(),
    )
    print(
        {
            "output": str(args.output.resolve()),
            "context_shape": list(encoding.tokens.shape),
            "num_goals": args.num_goals,
            "num_trajectories_per_goal": args.num_trajectories,
            "source_canonical": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

