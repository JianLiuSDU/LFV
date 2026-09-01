#!/usr/bin/env python3
"""Evaluate Direct, Transfer-only and confidence-fused Stage 2 inference.

The script is intentionally inference-only.  A source episode is exported as
an explicit motion-field memory, FGW transports both role fields to each
target episode, and the same checkpoint is evaluated with three conditions:
Direct (no memory), Transfer-only (the transported field is the bottleneck),
and Full (confidence-gated fusion of online and transported fields).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lfv.datasets.functional_motion import FunctionalMotionDataset
from lfv.evaluation.functional_motion import goal_metrics, trajectory_best_of_k_metrics
from lfv.models.functional_motion_generation import load_stage2_checkpoint
from lfv.models.functional_motion_generation.motion_field_transfer import (
    MotionFieldMemory,
    transport_motion_field,
)


def _batch(sample: dict) -> dict:
    return {
        key: value[None]
        if torch.is_tensor(value)
        else [value]
        for key, value in sample.items()
    }


def _metrics(samples, batch: dict) -> dict[str, float]:
    values = goal_metrics(samples.goals, batch["goal_pose9d"])
    values.update(
        trajectory_best_of_k_metrics(
            samples.trajectories, batch["trajectory_pose9d"]
        )
    )
    return {key: float(value) for key, value in values.items()}


def _transport(memory: MotionFieldMemory, sample: dict, *, alpha: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    fields: list[torch.Tensor] = []
    confidences: list[float] = []
    diagnostics: dict[str, object] = {}
    for role in ("manipulated", "reference"):
        result = transport_motion_field(
            getattr(memory, role + "_points"),
            getattr(memory, role + "_dino"),
            getattr(memory, role + "_field"),
            sample[role + "_points"].numpy(),
            sample[role + "_dino"].numpy(),
            node_count=min(64, len(memory.manipulated_points)),
            alpha=alpha,
            graph_neighbors=10,
            # RGB-D object clouds can contain several sparse components.  A
            # generous cap keeps the geodesic graph connected without adding
            # a task-specific geometric prior.
            graph_maximum_neighbors=64,
            edge_length_ratio=20.0,
            maximum_iterations=80,
        )
        fields.append(torch.from_numpy(result.target_field).float()[None])
        confidences.append(float(result.confidence))
        diagnostics[role] = {
            "confidence": float(result.confidence),
            "target_entropy": float(
                -np.sum(
                    result.target_field
                    * np.log(np.maximum(result.target_field, 1e-12))
                )
                / np.log(max(len(result.target_field), 2))
            ),
            "target_peak": float(result.target_field.max()),
            "solver": "FGW",
        }
    diagnostics["mean_confidence"] = float(np.mean(confidences))
    return fields[0], fields[1], diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-goals", type=int, default=2)
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _, _ = load_stage2_checkpoint(args.checkpoint, device=device, use_ema=True)
    model.eval()
    memory = MotionFieldMemory.load(args.memory)
    dataset = FunctionalMotionDataset(
        args.cache_root,
        args.split,
        shuffle_points=False,
        limit=args.limit,
    )
    conditions = ("direct", "transfer_only", "full")
    totals = {name: {} for name in conditions}
    counts = {name: 0 for name in conditions}
    diagnostics: dict[str, dict] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = _batch(sample)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        prior_m, prior_r, transfer_diag = _transport(memory, sample, alpha=args.alpha)
        prior = (prior_m.to(device), prior_r.to(device))
        diagnostics[str(sample["episode_id"])] = transfer_diag
        for condition in conditions:
            if condition == "direct":
                kwargs = {}
            elif condition == "transfer_only":
                model.encoder.motion_field_fusion_mode = "fixed"
                kwargs = {"motion_field_prior": prior, "motion_field_prior_weight": 1.0}
            else:
                model.encoder.motion_field_fusion_mode = "confidence"
                kwargs = {"motion_field_prior": prior, "motion_field_prior_weight": 0.5}
            generator = torch.Generator(device=device).manual_seed(args.seed)
            with torch.inference_mode():
                samples, _ = model.sample(
                    batch,
                    num_goal_samples=args.num_goals,
                    num_trajectory_samples=args.num_trajectories,
                    generator=generator,
                    **kwargs,
                )
            values = _metrics(samples, batch)
            for key, value in values.items():
                totals[condition][key] = totals[condition].get(key, 0.0) + value
            counts[condition] += 1
    results = {
        condition: {
            key: value / max(counts[condition], 1)
            for key, value in totals[condition].items()
        }
        for condition in conditions
    }
    direct = results["direct"]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "memory": str(args.memory.resolve()),
        "split": args.split,
        "episodes": len(dataset),
        "alpha": args.alpha,
        "num_goals": args.num_goals,
        "num_trajectories_per_goal": args.num_trajectories,
        "results": results,
        "delta_from_direct": {
            condition: {
                key: values[key] - direct[key]
                for key in direct
                if key in values
            }
            for condition, values in results.items()
            if condition != "direct"
        },
        "transfer_diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
