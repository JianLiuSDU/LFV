from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def render_topdown_grasp_summary(
    *,
    lift_overlay_path: str | Path,
    complete_heat_path: str | Path,
    selected_rgb_path: str | Path,
    selected_open3d_path: str | Path,
    grasp_report_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create the fixed 2x2 visual regression report for the 3D stage."""
    image_paths = [
        Path(lift_overlay_path),
        Path(complete_heat_path),
        Path(selected_rgb_path),
        Path(selected_open3d_path),
    ]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(Path(grasp_report_path).read_text(encoding="utf-8"))
    selected = report["selected"]
    collisions = selected["collision_part_ious"]
    max_collision = max(float(value) for value in collisions.values())
    titles = [
        "Transferred 2D heat on simulation RGB",
        "Heat completed on full manipulated-part surface",
        "Selected grasp in simulation camera",
        "Open3D: full heat + selected gripper",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for axis, path, title in zip(axes.flat, image_paths, titles, strict=True):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.suptitle(
        "Direction-constrained GraspNet result | "
        f"angle={selected['approach_to_desired_angle_deg']:.2f} deg | "
        f"pair width={selected['contact_pair_width_m'] * 1000:.2f} mm | "
        f"tip heat=({selected['left_tip_heat']:.3f}, "
        f"{selected['right_tip_heat']:.3f}) | "
        f"max collision IoU={max_collision:.3f}",
        fontsize=14,
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_instance_generalization_comparison(
    *,
    baseline_label: str,
    candidate_label: str,
    baseline_heat_path: str | Path,
    candidate_heat_path: str | Path,
    baseline_grasp_path: str | Path,
    candidate_grasp_path: str | Path,
    baseline_transfer_report_path: str | Path,
    candidate_transfer_report_path: str | Path,
    baseline_grasp_report_path: str | Path,
    candidate_grasp_report_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Compare affordance transfer and final grasp across two mug instances."""

    image_paths = [
        Path(baseline_heat_path),
        Path(candidate_heat_path),
        Path(baseline_grasp_path),
        Path(candidate_grasp_path),
    ]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    transfer_reports = [
        json.loads(Path(baseline_transfer_report_path).read_text(encoding="utf-8")),
        json.loads(Path(candidate_transfer_report_path).read_text(encoding="utf-8")),
    ]
    grasp_reports = [
        json.loads(Path(baseline_grasp_report_path).read_text(encoding="utf-8")),
        json.loads(Path(candidate_grasp_report_path).read_text(encoding="utf-8")),
    ]
    labels = [baseline_label, candidate_label]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for column, label in enumerate(labels):
        confidence = transfer_reports[column]["confidence"]
        selected = grasp_reports[column]["selected"]
        collision = max(
            float(value) for value in selected["collision_part_ious"].values()
        )
        for row, path in enumerate(
            [image_paths[column], image_paths[column + 2]]
        ):
            with Image.open(path) as image:
                axes[row, column].imshow(image.convert("RGB"))
            axes[row, column].axis("off")
        axes[0, column].set_title(
            f"{label} — transferred heat\n"
            f"confidence={confidence['global']:.3f}, "
            f"cycle={confidence['cycle']:.3f}",
            fontsize=13,
        )
        axes[1, column].set_title(
            f"{label} — selected grasp\n"
            f"angle={selected['approach_to_desired_angle_deg']:.2f} deg, "
            f"tip heat=({selected['left_tip_heat']:.3f}, "
            f"{selected['right_tip_heat']:.3f}), collision={collision:.3f}",
            fontsize=13,
        )
    fig.suptitle(
        "Same source demonstration, camera/layout protocol, and inference settings",
        fontsize=15,
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path
