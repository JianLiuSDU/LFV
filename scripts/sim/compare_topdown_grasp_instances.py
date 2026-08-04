#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.visualization import render_instance_generalization_comparison


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _instance_metrics(topdown_dir: Path) -> dict:
    transfer = _load_json(topdown_dir.parent / "transfer_report.json")
    grasp = _load_json(topdown_dir / "graspnet_full_contact_report.json")
    selected = grasp["selected"]
    return {
        "accepted": bool(transfer["accepted"]),
        "transfer_confidence": transfer["confidence"],
        "target_heat_location": transfer["diagnostics"]["target_heat_location"],
        "candidate_counts": {
            "decoded": int(grasp["num_decoded_grasps"]),
            "standard_collision_free": int(grasp["num_collision_free_grasps"]),
            "before_strict_collision": int(
                grasp["num_ranked_before_strict_collision"]
            ),
            "strict_collision_ranked": int(grasp["num_ranked_grasps"]),
        },
        "selected": {
            "final_score": float(selected["final_score"]),
            "approach_angle_deg": float(
                selected["approach_to_desired_angle_deg"]
            ),
            "left_tip_heat": float(selected["left_tip_heat"]),
            "right_tip_heat": float(selected["right_tip_heat"]),
            "contact_pair_width_m": float(selected["contact_pair_width_m"]),
            "normal_opposition": float(
                selected["contact_pair_geometry"]["normal_opposition"]
            ),
            "max_collision_iou": max(
                float(value)
                for value in selected["collision_part_ious"].values()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a fixed two-instance affordance/grasp comparison."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-label", default="YCB 025 mug")
    parser.add_argument("--candidate-label", default="new mug")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = render_instance_generalization_comparison(
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline_heat_path=baseline_dir / "transferred_heat_lift_overlay.png",
        candidate_heat_path=candidate_dir / "transferred_heat_lift_overlay.png",
        baseline_grasp_path=baseline_dir / "graspnet_selected_rgb_clean.png",
        candidate_grasp_path=candidate_dir / "graspnet_selected_rgb_clean.png",
        baseline_transfer_report_path=baseline_dir.parent / "transfer_report.json",
        candidate_transfer_report_path=candidate_dir.parent / "transfer_report.json",
        baseline_grasp_report_path=baseline_dir / "graspnet_full_contact_report.json",
        candidate_grasp_report_path=candidate_dir / "graspnet_full_contact_report.json",
        output_path=output_dir / "generalization_comparison.png",
    )
    baseline = _instance_metrics(baseline_dir)
    candidate = _instance_metrics(candidate_dir)
    report = {
        "comparison_scope": (
            "same source demonstration and camera/layout protocol; cup visual and "
            "geometry instance changed"
        ),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": {
            "transfer_global_confidence": (
                candidate["transfer_confidence"]["global"]
                - baseline["transfer_confidence"]["global"]
            ),
            "transfer_cycle": (
                candidate["transfer_confidence"]["cycle"]
                - baseline["transfer_confidence"]["cycle"]
            ),
            "approach_angle_deg": (
                candidate["selected"]["approach_angle_deg"]
                - baseline["selected"]["approach_angle_deg"]
            ),
            "final_score": (
                candidate["selected"]["final_score"]
                - baseline["selected"]["final_score"]
            ),
        },
        "limitations": [
            "The new asset has no manually annotated pixel-level handle ground truth.",
            "Transfer localization is therefore checked by confidence and fixed visual audit.",
            "Collision-free means static point-cloud collision filtering, not rollout success.",
        ],
        "visualization": str(image_path),
    }
    report_path = output_dir / "generalization_comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
