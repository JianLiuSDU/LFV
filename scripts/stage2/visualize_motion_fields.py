#!/usr/bin/env python3
"""Render explicit Stage 2 motion fields on RGB and observed point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lfv.datasets.functional_motion import FunctionalMotionDataset
from lfv.models.functional_motion_generation import load_stage2_checkpoint


def _first_rgb(source_root: Path, episode_id: str) -> np.ndarray:
    capture = cv2.VideoCapture(str(source_root / episode_id / "camera_0.mp4"))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read first RGB frame for {episode_id}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _overlay(
    rgb: np.ndarray,
    pixels_uv: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    heat = field / max(float(field.max()), 1e-12)
    colors = plt.get_cmap("turbo")(heat)[:, :3]
    canvas = rgb.astype(np.float32) / 255.0
    for (u, v), color, strength in zip(pixels_uv, colors, heat):
        u_int, v_int = int(u), int(v)
        if 0 <= v_int < canvas.shape[0] and 0 <= u_int < canvas.shape[1]:
            radius = 2 + int(round(3.0 * float(strength)))
            cv2.circle(
                canvas,
                (u_int, v_int),
                radius,
                tuple(float(value) for value in color),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
    return np.clip(canvas, 0.0, 1.0)


def _set_equal_3d(ax, points: np.ndarray) -> None:
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = max(float(np.ptp(points, axis=0).max()) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _normalized_entropy(field: np.ndarray) -> float:
    values = np.clip(field.astype(np.float64), 1e-12, None)
    values = values / max(float(values.sum()), 1e-12)
    return float(-(values * np.log(values)).sum() / np.log(len(values)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    dataset = FunctionalMotionDataset(cache_root, args.split, shuffle_points=False)
    index = next(
        (
            idx
            for idx, record in enumerate(dataset.records)
            if record["episode_id"] == args.episode_id
        ),
        None,
    )
    if index is None:
        raise ValueError(f"{args.episode_id} is not in split={args.split}")
    sample = dataset[index]
    artifact = Path(dataset.records[index]["artifact"])
    with np.load(artifact, allow_pickle=False) as data:
        manipulated_uv = np.asarray(data["manipulated_pixels_uv"])
        reference_uv = np.asarray(data["reference_pixels_uv"])

    batch = {
        key: value[None].to(args.device) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    model, _, payload = load_stage2_checkpoint(
        args.checkpoint, device=args.device, use_ema=True
    )
    with torch.no_grad():
        encoding = model.encode(batch, return_debug=True)
    if encoding.manipulated_motion_field is None:
        raise RuntimeError("Checkpoint does not produce an explicit motion field")
    manipulated_field = encoding.manipulated_motion_field[0].cpu().numpy()
    reference_field = encoding.reference_motion_field[0].cpu().numpy()
    manipulated_points = sample["manipulated_points"].numpy()
    reference_points = sample["reference_points"].numpy()
    rgb = _first_rgb(source_root, args.episode_id)

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    axes = [figure.add_subplot(2, 3, index + 1) for index in range(3)]
    axes += [figure.add_subplot(2, 3, 4, projection="3d")]
    axes += [figure.add_subplot(2, 3, 5, projection="3d")]
    axes += [figure.add_subplot(2, 3, 6)]
    axes[0].imshow(rgb)
    axes[0].set_title(f"{args.episode_id}: first-frame RGB")
    axes[1].imshow(_overlay(rgb, manipulated_uv, manipulated_field))
    axes[1].set_title("Manipulated-object motion field")
    axes[2].imshow(_overlay(rgb, reference_uv, reference_field))
    axes[2].set_title("Reference-object motion field")
    for axis in axes[:3]:
        axis.axis("off")

    for axis, points, field, title in (
        (axes[3], manipulated_points, manipulated_field, "Manipulated 3D field"),
        (axes[4], reference_points, reference_field, "Reference 3D field"),
    ):
        heat = field / max(float(field.max()), 1e-12)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=heat,
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            s=18,
        )
        axis.set_title(title)
        _set_equal_3d(axis, points)

    axes[5].hist(
        manipulated_field / max(float(manipulated_field.max()), 1e-12),
        bins=24,
        alpha=0.65,
        label="manipulated",
    )
    axes[5].hist(
        reference_field / max(float(reference_field.max()), 1e-12),
        bins=24,
        alpha=0.65,
        label="reference",
    )
    axes[5].set_title("Normalized field-value distribution")
    axes[5].legend()
    figure.suptitle(
        "Motion Functional Field | "
        f"epoch={payload.get('epoch', 'unknown')} | EMA weights",
        fontsize=15,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_id": args.episode_id,
        "split": args.split,
        "manipulated_entropy": _normalized_entropy(manipulated_field),
        "reference_entropy": _normalized_entropy(reference_field),
        "manipulated_peak_mass": float(manipulated_field.max()),
        "reference_peak_mass": float(reference_field.max()),
        "image": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
