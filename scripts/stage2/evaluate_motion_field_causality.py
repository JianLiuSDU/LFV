#!/usr/bin/env python3
"""Measure whether Stage 2 predictions causally depend on the learned field."""

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


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _evaluate(
    model,
    loader,
    *,
    device: torch.device,
    intervention: str | None,
    seed: int,
    num_goals: int,
    num_trajectories: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        batch = _to_device(batch, device)
        samples, encoding = model.sample(
            batch,
            num_goal_samples=num_goals,
            num_trajectory_samples=num_trajectories,
            generator=generator,
            motion_field_intervention=intervention,
        )
        metrics = goal_metrics(samples.goals, batch["goal_pose9d"])
        metrics.update(
            trajectory_best_of_k_metrics(
                samples.trajectories, batch["trajectory_pose9d"]
            )
        )
        for role, field in (
            ("manipulated", encoding.manipulated_motion_field),
            ("reference", encoding.reference_motion_field),
        ):
            if field is None:
                continue
            entropy = -(field * field.clamp_min(1e-12).log()).sum(dim=1)
            entropy = entropy / torch.log(
                torch.as_tensor(field.shape[1], device=device, dtype=field.dtype)
            )
            metrics[f"{role}_field_entropy"] = entropy.mean()
            metrics[f"{role}_field_peak"] = field.max(dim=1).values.mean()
        batch_size = int(batch["goal_pose9d"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
    return {key: value / count for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-goals", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _, _ = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=True
    )
    dataset = FunctionalMotionDataset(
        args.cache_root,
        args.split,
        shuffle_points=False,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_functional_motion,
    )
    results = {}
    for name, intervention in (
        ("learned", None),
        ("uniform", "uniform"),
        ("rolled", "roll"),
        ("drop_top", "drop_top"),
    ):
        results[name] = _evaluate(
            model,
            loader,
            device=device,
            intervention=intervention,
            seed=args.seed,
            num_goals=args.num_goals,
            num_trajectories=args.num_trajectories,
        )
    learned = results["learned"]
    deltas = {
        name: {
            key: values[key] - learned[key]
            for key in learned
            if key in values and "field_" not in key
        }
        for name, values in results.items()
        if name != "learned"
    }
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "episodes": len(dataset),
        "seed": args.seed,
        "num_goals": args.num_goals,
        "num_trajectories_per_goal": args.num_trajectories,
        "results": results,
        "delta_from_learned": deltas,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
