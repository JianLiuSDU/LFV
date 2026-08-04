#!/usr/bin/env python3
"""Create a reproducible quality manifest for drawer motion episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _episode_index(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="/media/ljian/lj/data_3d/drawer_lfv_v2",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-handle-pixels", type=int, default=100)
    parser.add_argument("--max-handle-pixels", type=int, default=2500)
    parser.add_argument("--max-handle-reference-ratio", type=float, default=0.15)
    parser.add_argument("--min-handle-contained-ratio", type=float, default=0.75)
    parser.add_argument("--require-trajectory", action="store_true")
    parser.add_argument("--min-final-motion-m", type=float, default=0.04)
    parser.add_argument("--max-final-motion-m", type=float, default=0.40)
    parser.add_argument("--max-path-to-displacement-ratio", type=float, default=25.0)
    parser.add_argument("--max-final-rotation-deg", type=float, default=25.0)
    args = parser.parse_args()

    root = Path(args.processed_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "drawer_quality_manifest.json"
    records = []
    accepted = []
    rejected = []
    for episode in sorted(root.glob("episode_*"), key=_episode_index):
        reasons = []
        handle_path = episode / "sam_mask" / "affordance_mask.npy"
        reference_path = episode / "target_sam_mask" / "target_mask.npy"
        metrics = {}
        if not handle_path.exists() or not reference_path.exists():
            reasons.append("missing_handle_or_reference_mask")
        else:
            handle = np.load(handle_path).astype(bool)
            reference = np.load(reference_path).astype(bool)
            handle_pixels = int(handle.sum())
            reference_pixels = int(reference.sum())
            ratio = float(handle_pixels / max(reference_pixels, 1))
            contained = float((handle & reference).sum() / max(handle_pixels, 1))
            metrics.update(
                handle_pixels=handle_pixels,
                reference_pixels=reference_pixels,
                handle_reference_area_ratio=ratio,
                handle_contained_ratio=contained,
            )
            if handle_pixels < args.min_handle_pixels:
                reasons.append("handle_mask_too_small")
            if handle_pixels > args.max_handle_pixels:
                reasons.append("handle_mask_too_large")
            if ratio > args.max_handle_reference_ratio:
                reasons.append("handle_mask_matches_whole_drawer")
            if contained < args.min_handle_contained_ratio:
                reasons.append("handle_mask_not_on_drawer")

        trajectory_path = episode / "se3_trajectory" / "dp_action_trajectory.npz"
        if args.require_trajectory:
            if not trajectory_path.exists():
                reasons.append("missing_trajectory")
            else:
                actions = np.load(trajectory_path)["actions_8d"]
                final_motion = float(np.linalg.norm(actions[-1, :3] - actions[0, :3]))
                path_length = float(
                    np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).sum()
                )
                path_ratio = float(path_length / max(final_motion, 1e-8))
                quat = np.asarray(actions[-1, 3:7], dtype=np.float64)
                quat /= max(float(np.linalg.norm(quat)), 1e-8)
                final_rotation_deg = float(
                    np.degrees(2.0 * np.arccos(np.clip(abs(quat[3]), 0.0, 1.0)))
                )
                metrics.update(
                    trajectory_steps=int(len(actions)),
                    final_motion_m=final_motion,
                    path_length_m=path_length,
                    path_to_displacement_ratio=path_ratio,
                    final_rotation_deg=final_rotation_deg,
                )
                if len(actions) != 64:
                    reasons.append("trajectory_not_64_steps")
                if final_motion < args.min_final_motion_m:
                    reasons.append("final_motion_too_small")
                if final_motion > args.max_final_motion_m:
                    reasons.append("final_motion_too_large")
                if path_ratio > args.max_path_to_displacement_ratio:
                    reasons.append("trajectory_too_jittery")
                if final_rotation_deg > args.max_final_rotation_deg:
                    reasons.append("final_rotation_too_large_for_prismatic_drawer")

        record = {
            "episode": episode.name,
            "accepted": not reasons,
            "reasons": reasons,
            "metrics": metrics,
        }
        records.append(record)
        (accepted if not reasons else rejected).append(episode.name)

    report = {
        "schema_version": 1,
        "processed_root": str(root),
        "criteria": {
            "min_handle_pixels": args.min_handle_pixels,
            "max_handle_pixels": args.max_handle_pixels,
            "max_handle_reference_ratio": args.max_handle_reference_ratio,
            "min_handle_contained_ratio": args.min_handle_contained_ratio,
            "require_trajectory": args.require_trajectory,
            "min_final_motion_m": args.min_final_motion_m,
            "max_final_motion_m": args.max_final_motion_m,
            "max_path_to_displacement_ratio": args.max_path_to_displacement_ratio,
            "max_final_rotation_deg": args.max_final_rotation_deg,
        },
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_episodes": accepted,
        "rejected_episodes": rejected,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("accepted_count", "rejected_count", "rejected_episodes")}, indent=2, ensure_ascii=False))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
