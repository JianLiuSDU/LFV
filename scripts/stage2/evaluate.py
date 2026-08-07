#!/usr/bin/env python3
"""Sample and evaluate a Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lfv.datasets.functional_motion import (
    FunctionalMotionDataset,
    collate_functional_motion,
)
from lfv.evaluation.functional_motion import goal_metrics, trajectory_best_of_k_metrics
from lfv.models.functional_motion_generation import load_stage2_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-goals", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _, _ = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=not args.no_ema
    )
    dataset = FunctionalMotionDataset(
        args.cache_root, args.split, shuffle_points=False
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_functional_motion,
    )
    aggregate: dict[str, list[float]] = {}
    for batch in loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        generator = torch.Generator(device=device).manual_seed(42)
        samples, _ = model.sample(
            batch,
            num_goal_samples=args.num_goals,
            num_trajectory_samples=args.num_trajectories,
            generator=generator,
        )
        metrics = goal_metrics(samples.goals, batch["goal_pose9d"])
        metrics.update(
            trajectory_best_of_k_metrics(
                samples.trajectories, batch["trajectory_pose9d"]
            )
        )
        for key, value in metrics.items():
            aggregate.setdefault(key, []).append(float(value))
    report = {key: sum(values) / len(values) for key, values in aggregate.items()}
    report.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "episodes": len(dataset),
            "num_goals": args.num_goals,
            "num_trajectories_per_goal": args.num_trajectories,
            "weights": "raw" if args.no_ema else "ema",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
