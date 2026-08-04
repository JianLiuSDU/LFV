#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.geometry import (
    ContactHeatPropagationConfig,
    UpperHandleOracleConfig,
    build_upper_handle_oracle_heat,
    propagate_contact_heat_to_opposite_surface,
    upper_handle_oracle_config_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a model-free upper-handle oracle contact field on the "
            "complete ManiSkill YCB mug surface."
        )
    )
    parser.add_argument(
        "--snapshot",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0_dataset_aligned/pouring_snapshot.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0_dataset_aligned/oracle_handle_upper_v2"
        ),
    )
    parser.add_argument("--handle-axis-x", type=float, default=1.0)
    parser.add_argument("--handle-axis-y", type=float, default=-1.0)
    parser.add_argument("--protrusion-min", type=float, default=0.045)
    parser.add_argument("--z-min", type=float, default=0.012)
    parser.add_argument("--z-max", type=float, default=0.030)
    parser.add_argument("--center-protrusion", type=float, default=0.057)
    parser.add_argument("--center-z", type=float, default=0.022)
    parser.add_argument("--sigma-protrusion", type=float, default=0.008)
    parser.add_argument("--sigma-lateral", type=float, default=0.010)
    parser.add_argument("--sigma-z", type=float, default=0.005)
    parser.add_argument("--min-width", type=float, default=0.004)
    parser.add_argument("--max-width", type=float, default=0.030)
    parser.add_argument("--min-antipodal-cos", type=float, default=0.55)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = np.load(snapshot_path)
    full_points_object = np.asarray(
        snapshot["full_points_object"],
        dtype=np.float32,
    )
    full_normals_object = np.asarray(
        snapshot["full_normals_object"],
        dtype=np.float32,
    )
    oracle_config = UpperHandleOracleConfig(
        handle_axis_xy=(args.handle_axis_x, args.handle_axis_y),
        protrusion_min_m=args.protrusion_min,
        z_min_m=args.z_min,
        z_max_m=args.z_max,
        center_protrusion_m=args.center_protrusion,
        center_z_m=args.center_z,
        sigma_protrusion_m=args.sigma_protrusion,
        sigma_lateral_m=args.sigma_lateral,
        sigma_z_m=args.sigma_z,
    )
    oracle = build_upper_handle_oracle_heat(
        full_points_object,
        config=oracle_config,
    )
    propagation_config = ContactHeatPropagationConfig(
        projection_radius=0.001,
        seed_quantile=0.65,
        max_seed_points=256,
        min_contact_width=args.min_width,
        max_contact_width=args.max_width,
        min_antipodal_cos=args.min_antipodal_cos,
        min_normal_opposition_cos=args.min_antipodal_cos,
        local_spread_radius=0.004,
        local_spread_sigma=0.002,
        opposite_pairs_per_seed=3,
        min_pair_score=0.03,
        require_hidden_opposite=False,
    )
    propagation = propagate_contact_heat_to_opposite_surface(
        full_points_object,
        oracle.heat,
        full_points_object,
        full_normals_object,
        config=propagation_config,
    )
    pair_array = np.asarray(
        [
            [
                pair.visible_index,
                pair.opposite_index,
                pair.score,
                pair.width_m,
                pair.visible_alignment,
                pair.opposite_alignment,
                pair.normal_opposition,
                pair.opposite_visible_distance_m,
                float(pair.opposite_is_hidden),
            ]
            for pair in propagation.pairs
        ],
        dtype=np.float32,
    ).reshape(-1, 9)
    output_path = output_dir / "oracle_handle_contact.npz"
    np.savez_compressed(
        output_path,
        full_points_object=full_points_object,
        full_normals_object=full_normals_object,
        full_points_camera=snapshot["full_points_camera"],
        full_normals_camera=snapshot["full_normals_camera"],
        oracle_semantic_heat=oracle.heat,
        oracle_candidate_mask=oracle.candidate_mask.astype(np.uint8),
        projected_visible_heat=propagation.visible_heat,
        pair_visible_heat=propagation.pair_visible_heat,
        opposite_heat=propagation.opposite_heat,
        full_heat=propagation.full_heat,
        antipodal_pairs=pair_array,
        scene_points_camera=snapshot["scene_points_camera"],
        scene_colors=snapshot["scene_colors"],
        scene_is_cup=snapshot["scene_is_cup"],
        T_object_to_camera=snapshot["T_object_to_camera"],
        T_object_to_world=snapshot["T_object_to_world"],
        T_world_to_camera=snapshot["T_world_to_camera"],
    )
    pair_widths = pair_array[:, 3] if len(pair_array) else np.zeros(0)
    report = {
        "source": "complete-geometry upper-handle oracle; Contact model bypassed",
        "snapshot": str(snapshot_path),
        "oracle_config": upper_handle_oracle_config_dict(oracle_config),
        "oracle": oracle.summary(),
        "propagation_config": propagation_config.__dict__,
        "propagation": propagation.summary(),
        "pair_width_m": {
            "min": float(pair_widths.min()) if len(pair_widths) else 0.0,
            "median": float(np.median(pair_widths)) if len(pair_widths) else 0.0,
            "max": float(pair_widths.max(initial=0.0)),
        },
        "output": str(output_path),
    }
    report_path = output_dir / "oracle_handle_contact_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
