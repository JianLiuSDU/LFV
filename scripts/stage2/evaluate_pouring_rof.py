#!/usr/bin/env python3
"""Evaluate pouring success from an inferred object trajectory and explicit geometry.

The geometry file is intentionally separate from model inputs.  It is produced
by the simulator/asset adapter and contains rim samples and bowl opening
parameters, so the metric cannot silently fall back to the cup center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lfv.evaluation.functional_motion import pouring_success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rof", type=float, default=0.20)
    parser.add_argument("--height-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    with np.load(args.prediction, allow_pickle=False) as prediction:
        poses = np.asarray(
            prediction["pred_object_poses_world"], dtype=np.float32
        )
    with np.load(args.geometry, allow_pickle=False) as geometry:
        rim_object = np.asarray(geometry["rim_points_object"], dtype=np.float32)
        center = np.asarray(geometry["opening_center_world"], dtype=np.float32)
        normal = np.asarray(geometry["opening_normal_world"], dtype=np.float32)
        radius = float(np.asarray(geometry["opening_radius_m"]).reshape(()))
        collision_free = (
            np.asarray(geometry["collision_free"], dtype=bool)
            if "collision_free" in geometry
            else None
        )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("pred_object_poses_world must have shape [T,4,4]")
    metrics = []
    for pose in poses:
        rim_h = np.concatenate(
            (rim_object, np.ones((rim_object.shape[0], 1), dtype=np.float32)),
            axis=1,
        )
        rim_world = (pose @ rim_h.T).T[:, :3]
        metrics.append(
            pouring_success(
                rim_world,
                center,
                normal,
                radius,
                min_rof=args.min_rof,
                height_tolerance_m=args.height_tolerance,
            )
        )
    final = metrics[-1]
    collision_ok = bool(collision_free is None or collision_free.all())
    report = {
        "prediction": str(args.prediction.resolve()),
        "geometry": str(args.geometry.resolve()),
        "frames": len(metrics),
        "min_rof": args.min_rof,
        "height_tolerance_m": args.height_tolerance,
        "final": final,
        "max_rim_over_opening_fraction": float(
            max(item["rim_over_opening_fraction"] for item in metrics)
        ),
        "collision_free": collision_ok,
        "success": bool(final["success"] and collision_ok),
        "per_frame": metrics,
        "geometry_contract": {
            "rim_points_object": list(rim_object.shape),
            "opening_center_world": list(center.shape),
            "opening_normal_world": list(normal.shape),
            "opening_radius_m": radius,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

