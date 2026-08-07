#!/usr/bin/env python3
"""Audit trajectory frequency/phase fidelity for a Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lfv.datasets.functional_motion import (
    FunctionalMotionDataset,
    collate_functional_motion,
)
from lfv.evaluation.functional_motion import trajectory_spectrum_summary
from lfv.models.functional_motion_generation import load_stage2_checkpoint


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _sample_gt_goal(
    model,
    batch: dict,
    generator: torch.Generator,
    inference_steps: int | None,
) -> torch.Tensor:
    context = model.encode(batch).tokens
    return model.trajectory_diffuser.sample(
        context,
        batch["goal_pose9d"][:, None],
        model.normalizer,
        num_samples_per_goal=1,
        generator=generator,
        inference_steps=inference_steps,
    )[:, 0, 0]


@torch.no_grad()
def _sample_predicted_goal(
    model,
    batch: dict,
    generator: torch.Generator,
    inference_steps: int | None,
) -> torch.Tensor:
    samples, _ = model.sample(
        batch,
        num_goal_samples=1,
        num_trajectory_samples=1,
        generator=generator,
        goal_inference_steps=inference_steps,
        trajectory_inference_steps=inference_steps,
    )
    return samples.trajectories[:, 0, 0]


def _plot(output: Path, prediction: np.ndarray, target: np.ndarray, spectra: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for sample in range(min(6, prediction.shape[0])):
        axes[0, 0].plot(
            prediction[sample, :, 0], prediction[sample, :, 2], alpha=0.75
        )
        axes[0, 1].plot(target[sample, :, 0], target[sample, :, 2], alpha=0.75)
    axes[0, 0].set_title("Prediction: translation X-Z")
    axes[0, 1].set_title("Ground truth: translation X-Z")
    for axis in axes[0]:
        axis.set_xlabel("x / m")
        axis.set_ylabel("z / m")
        axis.set_aspect("equal", adjustable="datalim")

    axes[1, 0].semilogy(
        spectra["position_frequency"],
        spectra["position_magnitude_target"] + 1e-12,
        label="GT",
    )
    axes[1, 0].semilogy(
        spectra["position_frequency"],
        spectra["position_magnitude_prediction"] + 1e-12,
        label="prediction",
    )
    axes[1, 0].set_title("DCT of endpoint-detrended position")
    axes[1, 0].set_xlabel("frequency bin")
    axes[1, 0].legend()
    axes[1, 1].semilogy(
        spectra["velocity_frequency"],
        spectra["velocity_magnitude_target"] + 1e-12,
        label="GT",
    )
    axes[1, 1].semilogy(
        spectra["velocity_frequency"],
        spectra["velocity_magnitude_prediction"] + 1e-12,
        label="prediction",
    )
    axes[1, 1].set_title("DCT of frame-to-frame velocity")
    axes[1, 1].set_xlabel("frequency bin")
    axes[1, 1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--goal-source", choices=("gt", "predicted"), default="gt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config, payload = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=not args.no_ema
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
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[str] = []
    for batch_index, batch in enumerate(loader):
        batch = _to_device(batch, device)
        generator = torch.Generator(device=device).manual_seed(
            args.seed + batch_index
        )
        if args.goal_source == "gt":
            sampled = _sample_gt_goal(
                model, batch, generator, args.inference_steps
            )
        else:
            sampled = _sample_predicted_goal(
                model, batch, generator, args.inference_steps
            )
        scale = batch["scene_scale"].reshape(-1, 1, 1)
        predictions.append((sampled[..., :3] * scale).cpu().numpy())
        targets.append((batch["trajectory_pose9d"][..., :3] * scale).cpu().numpy())
        episode_ids.extend(batch["episode_id"])

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    metrics, spectra = trajectory_spectrum_summary(prediction, target)
    model_config = config.get("model", {})
    report: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "weights": "raw" if args.no_ema else "ema",
        "split": args.split,
        "episodes": len(dataset),
        "goal_source": args.goal_source,
        "seed": args.seed,
        "trajectory_position_encoding": model.trajectory_diffuser.decoder.position_encoding,
        "trajectory_architecture": {
            "layers": model_config.get("trajectory_layers"),
            "hidden_dim": model_config.get("hidden_dim"),
            "heads": model_config.get("decoder_heads"),
        },
        "metrics": metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spectrum_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "spectrum_samples.npz",
        prediction=prediction,
        target=target,
        episode_id=np.asarray(episode_ids),
        **spectra,
    )
    _plot(output_dir / "spectrum_comparison.png", prediction, target, spectra)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
