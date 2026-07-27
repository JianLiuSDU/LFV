#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = [
    "points_camera",
    "points_object_m",
    "points_object_norm",
    "pixels_uv",
    "normals_camera",
    "contact_evidence",
    "contact_heat",
    "heatmap_2d",
    "object_mask_anchor",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LFV contact_field outputs for one episode.")
    parser.add_argument("episode")
    parser.add_argument("--subdir", default="contact_field")
    parser.add_argument("--output-dir", default=None, help="Validate this output directory directly.")
    args = parser.parse_args()

    ep = Path(args.episode)
    out_dir = Path(args.output_dir) if args.output_dir else ep / args.subdir
    npz_path = out_dir / "contact_field.npz"
    meta_path = out_dir / "contact_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    data = np.load(npz_path)
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    n = data["points_camera"].shape[0]
    for key in ["points_object_m", "points_object_norm", "pixels_uv", "normals_camera", "contact_evidence", "contact_heat"]:
        if data[key].shape[0] != n:
            raise ValueError(f"{key} first dim {data[key].shape[0]} != point count {n}")
    for key in ["points_camera", "points_object_m", "points_object_norm", "normals_camera", "contact_evidence", "contact_heat", "heatmap_2d"]:
        arr = data[key]
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} contains NaN/Inf")
    heat = data["contact_heat"]
    if float(np.min(heat)) < -1e-6 or float(np.max(heat)) > 1.000001:
        raise ValueError(f"contact_heat outside [0,1]: min={heat.min()} max={heat.max()}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    viz_dir = out_dir / "viz"
    viz = sorted(p.name for p in viz_dir.glob("*.png")) if viz_dir.exists() else []
    print(
        json.dumps(
            {
                "episode": str(ep),
                "point_count": int(n),
                "quality": meta.get("quality"),
                "seed_count": meta.get("seed_count"),
                "heat_max": float(np.max(heat)),
                "heat_mean": float(np.mean(heat)),
                "viz_files": viz,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
