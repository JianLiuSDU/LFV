#!/usr/bin/env python3
"""Create a compact comparison plot from completed Stage 2 ablation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help="NAME=/absolute/stage/output/directory",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for specification in args.stage:
        name, raw_root = specification.split("=", 1)
        root = Path(raw_root)
        standard = _read(root / "test_metrics.json")
        gt = _read(root / "spectrum_gt_goal" / "spectrum_report.json")["metrics"]
        predicted = _read(
            root / "spectrum_predicted_goal" / "spectrum_report.json"
        )["metrics"]
        rows.append(
            {
                "stage": name,
                "trajectory_top1_translation_m": standard[
                    "trajectory_top1_translation_m"
                ],
                "trajectory_top1_rotation_deg": standard[
                    "trajectory_top1_rotation_deg"
                ],
                "gt_goal_mid_position_retention": gt[
                    "position_mid_energy_retention"
                ],
                "gt_goal_mid_position_cosine": gt[
                    "position_mid_coefficient_cosine"
                ],
                "gt_goal_mid_velocity_retention": gt[
                    "velocity_mid_energy_retention"
                ],
                "gt_goal_mid_velocity_cosine": gt[
                    "velocity_mid_coefficient_cosine"
                ],
                "gt_goal_curvature_frame_error": gt[
                    "dominant_curvature_frame_error"
                ],
                "gt_goal_first_step_ratio": gt["first_step_prediction_m"]
                / max(gt["first_step_target_m"], 1e-12),
                "gt_goal_path_length_ratio": gt["path_length_ratio"],
                "predicted_goal_mid_position_retention": predicted[
                    "position_mid_energy_retention"
                ],
            }
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    names = [row["stage"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].bar(
        x,
        [100 * row["trajectory_top1_translation_m"] for row in rows],
        color="#4C78A8",
    )
    axes[0, 0].set_ylabel("cm")
    axes[0, 0].set_title("Test top-1 trajectory translation (lower is better)")
    axes[0, 1].bar(
        x - width / 2,
        [row["gt_goal_mid_position_retention"] for row in rows],
        width,
        label="energy retention",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["gt_goal_mid_position_cosine"] for row in rows],
        width,
        label="coefficient cosine",
    )
    axes[0, 1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0, 1].set_title("GT-goal detrended position: mid band")
    axes[0, 1].legend()
    axes[1, 0].bar(
        x,
        [row["gt_goal_curvature_frame_error"] for row in rows],
        color="#F58518",
    )
    axes[1, 0].set_ylabel("frames")
    axes[1, 0].set_title("Dominant curvature-frame error (lower is better)")
    axes[1, 1].bar(
        x - width / 2,
        [row["gt_goal_first_step_ratio"] for row in rows],
        width,
        label="first-step ratio",
    )
    axes[1, 1].bar(
        x + width / 2,
        [row["gt_goal_path_length_ratio"] for row in rows],
        width,
        label="path-length ratio",
    )
    axes[1, 1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_title("Local boundary and global path ratios (target=1)")
    axes[1, 1].legend()
    for axis in axes.reshape(-1):
        axis.set_xticks(x, names, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "ablation_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
