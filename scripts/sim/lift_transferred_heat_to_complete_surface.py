#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.geometry import (
    ContactHeatPropagationConfig,
    propagate_contact_heat_to_opposite_surface,
)
from lfv.lifting import lift_image_heat_to_camera


def _first(arrays: dict[str, np.ndarray], *keys: str) -> np.ndarray:
    for key in keys:
        if key in arrays:
            return arrays[key]
    raise KeyError(f"Snapshot is missing all compatible keys: {keys}")


def _pair_array(result) -> np.ndarray:
    return np.asarray(
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
            for pair in result.pairs
        ],
        dtype=np.float32,
    ).reshape(-1, 9)


def _save_lift_overlay(
    output_path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    threshold: float,
) -> None:
    normalized = np.clip(heatmap, 0.0, 1.0)
    heat_u8 = np.round(normalized * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    alpha = (0.10 + 0.65 * np.sqrt(normalized))[..., None]
    active = (mask.astype(bool) & (normalized >= threshold))[..., None]
    canvas = np.where(
        active,
        canvas.astype(np.float32) * (1.0 - alpha)
        + heat_bgr.astype(np.float32) * alpha,
        canvas,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lift a transferred 2D contact heatmap through the aligned "
            "ManiSkill depth image and complete it with antipodal surface pairs."
        )
    )
    parser.add_argument("--transfer-result", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visible-heat-threshold", type=float, default=0.15)
    parser.add_argument("--projection-radius", type=float, default=0.006)
    parser.add_argument("--seed-quantile", type=float, default=0.65)
    parser.add_argument("--max-seed-points", type=int, default=256)
    parser.add_argument("--min-contact-width", type=float, default=0.004)
    parser.add_argument("--max-contact-width", type=float, default=0.030)
    parser.add_argument("--min-antipodal-cos", type=float, default=0.55)
    parser.add_argument("--min-normal-opposition-cos", type=float, default=0.65)
    parser.add_argument("--local-spread-radius", type=float, default=0.004)
    parser.add_argument("--local-spread-sigma", type=float, default=0.002)
    parser.add_argument("--local-normal-cos", type=float, default=0.55)
    parser.add_argument("--opposite-pairs-per-seed", type=int, default=3)
    parser.add_argument("--min-pair-score", type=float, default=0.03)
    parser.add_argument("--hidden-distance", type=float, default=0.006)
    parser.add_argument("--require-hidden-opposite", action="store_true")
    args = parser.parse_args()

    transfer_path = Path(args.transfer_result).expanduser().resolve()
    snapshot_path = Path(args.snapshot).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(transfer_path) as transfer:
        target_heatmap = np.asarray(transfer["target_heatmap"], dtype=np.float32)
    with np.load(snapshot_path) as snapshot:
        snapshot_arrays = {key: np.asarray(snapshot[key]) for key in snapshot.files}

    manipulated_mask = _first(snapshot_arrays, "manipulated_mask", "cup_mask")
    full_points_local = _first(
        snapshot_arrays, "full_manipulated_points_local", "full_points_object"
    )
    full_normals_local = _first(
        snapshot_arrays, "full_manipulated_normals_local", "full_normals_object"
    )
    full_points_camera = _first(
        snapshot_arrays, "full_manipulated_points_camera", "full_points_camera"
    )
    full_normals_camera = _first(
        snapshot_arrays, "full_manipulated_normals_camera", "full_normals_camera"
    )
    T_object_to_world = _first(
        snapshot_arrays, "T_manipulated_to_world", "T_object_to_world"
    )
    T_object_to_camera = _first(
        snapshot_arrays, "T_manipulated_to_camera", "T_object_to_camera"
    )
    scene_points_camera = _first(
        snapshot_arrays, "complete_scene_points_camera", "scene_points_camera"
    )
    scene_colors = _first(
        snapshot_arrays, "complete_scene_colors", "scene_colors"
    )
    scene_is_manipulated = _first(
        snapshot_arrays, "complete_scene_is_manipulated", "scene_is_cup"
    ).astype(bool)

    lifted = lift_image_heat_to_camera(
        target_heatmap,
        snapshot_arrays["depth_m"],
        manipulated_mask,
        snapshot_arrays["intrinsic_cv"],
        heat_threshold=args.visible_heat_threshold,
    )
    propagation_config = ContactHeatPropagationConfig(
        projection_radius=args.projection_radius,
        seed_quantile=args.seed_quantile,
        max_seed_points=args.max_seed_points,
        min_contact_width=args.min_contact_width,
        max_contact_width=args.max_contact_width,
        min_antipodal_cos=args.min_antipodal_cos,
        min_normal_opposition_cos=args.min_normal_opposition_cos,
        local_spread_radius=args.local_spread_radius,
        local_spread_sigma=args.local_spread_sigma,
        local_normal_cos=args.local_normal_cos,
        opposite_pairs_per_seed=args.opposite_pairs_per_seed,
        min_pair_score=args.min_pair_score,
        hidden_distance=args.hidden_distance,
        require_hidden_opposite=args.require_hidden_opposite,
    )
    propagation = propagate_contact_heat_to_opposite_surface(
        lifted.points_camera,
        lifted.heat,
        full_points_camera,
        full_normals_camera,
        config=propagation_config,
    )
    pairs = _pair_array(propagation)
    if not len(pairs):
        raise RuntimeError(
            "The transferred visible heat produced no antipodal pair on the complete manipulated part."
        )

    output_path = output_dir / "transferred_contact_3d.npz"
    np.savez_compressed(
        output_path,
        target_heatmap_2d=target_heatmap,
        visible_pixels_uv=lifted.pixels_uv,
        visible_points_camera=lifted.points_camera,
        visible_heat=lifted.heat,
        visible_heat_raw=lifted.raw_heat,
        full_points_object=full_points_local,
        full_normals_object=full_normals_local,
        full_points_camera=full_points_camera,
        full_normals_camera=full_normals_camera,
        projected_visible_heat=propagation.visible_heat,
        pair_visible_heat=propagation.pair_visible_heat,
        opposite_heat=propagation.opposite_heat,
        full_heat=propagation.full_heat,
        antipodal_pairs=pairs,
        scene_points_camera=scene_points_camera,
        scene_colors=scene_colors,
        scene_is_manipulated=scene_is_manipulated,
        # Legacy alias retained for the existing GraspNet consumer.
        scene_is_cup=scene_is_manipulated,
        T_object_to_camera=T_object_to_camera,
        T_object_to_world=T_object_to_world,
        T_manipulated_to_camera=T_object_to_camera,
        T_manipulated_to_world=T_object_to_world,
        T_world_to_camera=snapshot_arrays["T_world_to_camera"],
    )
    overlay_path = output_dir / "transferred_heat_lift_overlay.png"
    _save_lift_overlay(
        overlay_path,
        snapshot_arrays["rgb"],
        manipulated_mask,
        target_heatmap,
        args.visible_heat_threshold,
    )
    pair_widths = pairs[:, 3]
    report = {
        "stage": "transferred_2d_heat_to_complete_antipodal_surface",
        "transfer_result": str(transfer_path),
        "snapshot": str(snapshot_path),
        "visible_heat_threshold": args.visible_heat_threshold,
        "lifting": lifted.summary(),
        "propagation_config": propagation_config.__dict__,
        "propagation": propagation.summary(),
        "pair_width_m": {
            "min": float(pair_widths.min()),
            "median": float(np.median(pair_widths)),
            "max": float(pair_widths.max()),
        },
        "outputs": {
            "contact_3d": str(output_path),
            "lift_overlay": str(overlay_path),
        },
    }
    report_path = output_dir / "transferred_contact_3d_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
