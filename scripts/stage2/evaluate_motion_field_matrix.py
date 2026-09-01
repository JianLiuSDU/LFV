#!/usr/bin/env python3
"""Paired Stage-2 checkpoint and motion-field intervention evaluation.

This is a task-level evaluation harness, not a training script.  Every
checkpoint/intervention sees the same episode order and per-episode seed.  The
output contains per-episode values plus bootstrap confidence intervals so a
small change in denoising loss cannot be mistaken for a task improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from lfv.datasets.functional_motion import FunctionalMotionDataset
from lfv.evaluation.functional_motion import (
    goal_metrics,
    trajectory_best_of_k_metrics,
)
from lfv.models.functional_motion_generation import load_stage2_checkpoint


def _to_device(sample: dict, device: torch.device) -> dict:
    return {
        key: value[None].to(device)
        if torch.is_tensor(value)
        else value
        for key, value in sample.items()
    }


def _float_metrics(samples, batch: dict) -> dict[str, float]:
    values = goal_metrics(samples.goals, batch["goal_pose9d"])
    values.update(
        trajectory_best_of_k_metrics(
            samples.trajectories, batch["trajectory_pose9d"]
        )
    )
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def _bootstrap_mean(values: Iterable[float], *, seed: int, rounds: int) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(rounds, array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _bootstrap_delta(
    values: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    rounds: int,
) -> dict[str, float]:
    delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    result = _bootstrap_mean(delta, seed=seed, rounds=rounds)
    result["delta_mean"] = result.pop("mean")
    return result


def _evaluate_checkpoint(
    checkpoint: Path,
    dataset: FunctionalMotionDataset,
    *,
    device: torch.device,
    interventions: tuple[str | None, ...],
    seeds: tuple[int, ...],
    num_goals: int,
    num_trajectories: int,
    bootstrap_rounds: int,
) -> dict:
    model, config, payload = load_stage2_checkpoint(
        checkpoint, device=device, use_ema=True
    )
    model.eval()
    conditions: dict[str, list[dict[str, float]]] = {
        "learned" if item is None else item: [] for item in interventions
    }
    episode_ids = []
    for episode_index in range(len(dataset)):
        sample = dataset[episode_index]
        episode_id = str(sample.get("episode_id", episode_index))
        episode_ids.append(episode_id)
        batch = _to_device(sample, device)
        for intervention in interventions:
            name = "learned" if intervention is None else intervention
            per_seed: list[dict[str, float]] = []
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(
                    int(seed + 100003 * episode_index)
                )
                with torch.inference_mode():
                    samples, encoding = model.sample(
                        batch,
                        num_goal_samples=num_goals,
                        num_trajectory_samples=num_trajectories,
                        generator=generator,
                        motion_field_intervention=intervention,
                    )
                values = _float_metrics(samples, batch)
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
                    values[f"{role}_field_entropy"] = float(entropy.mean().cpu())
                    values[f"{role}_field_peak"] = float(field.max(dim=1).values.mean().cpu())
                per_seed.append(values)
            keys = per_seed[0].keys()
            conditions[name].append(
                {
                    key: float(np.mean([row[key] for row in per_seed]))
                    for key in keys
                }
            )
    summary: dict[str, dict] = {}
    for name, rows in conditions.items():
        metric_names = rows[0].keys() if rows else ()
        summary[name] = {
            metric: _bootstrap_mean(
                [row[metric] for row in rows],
                seed=17,
                rounds=bootstrap_rounds,
            )
            for metric in metric_names
        }
    learned_rows = conditions["learned"]
    deltas: dict[str, dict] = {}
    for name, rows in conditions.items():
        if name == "learned":
            continue
        common = set(learned_rows[0]).intersection(rows[0]) if rows else set()
        deltas[name] = {
            metric: _bootstrap_delta(
                np.asarray([row[metric] for row in rows]),
                np.asarray([row[metric] for row in learned_rows]),
                seed=31,
                rounds=bootstrap_rounds,
            )
            for metric in sorted(common)
            if "field_" not in metric
        }
    return {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": int(payload.get("epoch", -1)),
        "model_config": config.get("model", {}),
        "episodes": episode_ids,
        "num_goals": num_goals,
        "num_trajectories_per_goal": num_trajectories,
        "seeds": list(seeds),
        "summary": summary,
        "delta_from_learned": deltas,
        "per_episode": conditions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeatable checkpoint spec, for example v2=/path/best.pt",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-goals", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = FunctionalMotionDataset(
        args.cache_root,
        args.split,
        shuffle_points=False,
        limit=args.limit,
    )
    interventions = (
        None,
        "uniform",
        "roll",
        "complement",
        "keep_top_05",
        "keep_top_10",
        "keep_top_20",
        "drop_top_05",
        "drop_top_10",
        "drop_top_20",
        "drop_top_30",
    )
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    checkpoints = {}
    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError("--checkpoint must have NAME=PATH format")
        name, path = spec.split("=", 1)
        checkpoints[name] = _evaluate_checkpoint(
            Path(path),
            dataset,
            device=device,
            interventions=interventions,
            seeds=seeds,
            num_goals=args.num_goals,
            num_trajectories=args.num_trajectories,
            bootstrap_rounds=args.bootstrap_rounds,
        )
    report = {
        "cache_root": str(args.cache_root.resolve()),
        "split": args.split,
        "episodes": len(dataset),
        "interventions": ["learned" if item is None else item for item in interventions],
        "paired_seed_policy": "seed + 100003 * episode_index; reset for every intervention/checkpoint",
        "bootstrap_rounds": args.bootstrap_rounds,
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
