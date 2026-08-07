#!/usr/bin/env python3
"""Measure Goal/Trajectory gradient interaction on the shared scene encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lfv.datasets.functional_motion import (
    FunctionalMotionDataset,
    collate_functional_motion,
)
from lfv.models.functional_motion_generation import load_stage2_checkpoint


def _gradient_stats(first, second) -> tuple[float, float, float]:
    dot = torch.zeros((), device=first[0].device)
    first_square = torch.zeros_like(dot)
    second_square = torch.zeros_like(dot)
    for left, right in zip(first, second):
        dot = dot + torch.sum(left * right)
        first_square = first_square + torch.sum(left.square())
        second_square = second_square + torch.sum(right.square())
    first_norm = first_square.sqrt()
    second_norm = second_square.sqrt()
    cosine = dot / (first_norm * second_norm).clamp_min(1e-12)
    return float(first_norm), float(second_norm), float(cosine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _, payload = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=not args.no_ema
    )
    dataset = FunctionalMotionDataset(
        args.cache_root, args.split, shuffle_points=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_functional_motion,
    )
    parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    goal_norms: list[float] = []
    trajectory_norms: list[float] = []
    cosines: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        torch.manual_seed(args.seed + batch_index)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        context = model.encode(batch).tokens
        goal_loss = model.goal_diffuser.compute_loss(
            context, batch["goal_pose9d"], model.normalizer
        )["goal_total"]
        trajectory_loss = model.trajectory_diffuser.compute_loss(
            context,
            batch["goal_pose9d"],
            batch["trajectory_pose9d"],
            model.normalizer,
        )["trajectory_total"]
        goal_gradient = torch.autograd.grad(
            goal_loss, parameters, retain_graph=True
        )
        trajectory_gradient = torch.autograd.grad(trajectory_loss, parameters)
        goal_norm, trajectory_norm, cosine = _gradient_stats(
            goal_gradient, trajectory_gradient
        )
        goal_norms.append(goal_norm)
        trajectory_norms.append(trajectory_norm)
        cosines.append(cosine)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "weights": "raw" if args.no_ema else "ema",
        "split": args.split,
        "num_batches": len(cosines),
        "goal_gradient_norm_mean": float(np.mean(goal_norms)),
        "trajectory_gradient_norm_mean": float(np.mean(trajectory_norms)),
        "trajectory_to_goal_norm_ratio": float(
            np.mean(np.asarray(trajectory_norms) / np.maximum(goal_norms, 1e-12))
        ),
        "gradient_cosine_mean": float(np.mean(cosines)),
        "gradient_cosine_std": float(np.std(cosines)),
        "gradient_cosine_min": float(np.min(cosines)),
        "negative_cosine_fraction": float(np.mean(np.asarray(cosines) < 0.0)),
        "per_batch": {
            "goal_gradient_norm": goal_norms,
            "trajectory_gradient_norm": trajectory_norms,
            "gradient_cosine": cosines,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
