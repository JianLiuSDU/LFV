#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.utils.config import load_config


def _episode_name(value: str) -> str:
    return value if value.startswith("episode_") else f"episode_{value}"


def _count_frame_files(path: pathlib.Path) -> int:
    return len(list(path.glob("frame_*.npy"))) if path.exists() else 0


def _load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_episode(ep: pathlib.Path) -> dict:
    timing = _load_json(ep / "contact_timing" / "contact_timing.json")
    heat_meta = _load_json(ep / "contact_heatmap" / "contact_heatmap_meta.json")
    dino_meta = _load_json(ep / "dinov2_features" / "dinov2_meta.json")

    result = {
        "episode": ep.name,
        "hand_bbox": _count_frame_files(ep / "hand_bbox"),
        "hand_mask": _count_frame_files(ep / "hand_mask"),
        "timing_quality": timing.get("quality") if timing else "missing",
        "anchor_frame": timing.get("anchor_frame") if timing else None,
        "contact_start": timing.get("contact_start") if timing else None,
        "contact_end": timing.get("contact_end") if timing else None,
        "dinov2_points": dino_meta.get("point_count") if dino_meta else None,
        "heat_exists": (ep / "contact_heatmap" / "contact_heatmap.npz").exists(),
        "seed_count": heat_meta.get("seed_count") if heat_meta else None,
        "heat_area_ratio": heat_meta.get("heat_area_ratio") if heat_meta else None,
    }

    heat_path = ep / "contact_heatmap" / "contact_heatmap.npz"
    if heat_path.exists():
        try:
            data = np.load(heat_path)
            result["contact_heat_shape"] = tuple(data["contact_heat"].shape)
            result["contact_heat_max"] = float(np.max(data["contact_heat"]))
        except Exception as exc:
            result["heat_load_error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize hand pouring contact preprocessing outputs.")
    parser.add_argument("--config", default="configs/pipeline/hand_pouring.yaml")
    parser.add_argument("--episode", action="append", default=[], help="Episode id/name. Can be repeated.")
    parser.add_argument("--show-bad", action="store_true", help="Print episodes with missing/reject/empty outputs.")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    selected = [_episode_name(ep) for ep in args.episode] if args.episode else None
    episodes = iter_processed_episodes(cfg.paths.processed_root, selected)

    rows = [summarize_episode(ep) for ep in episodes]
    timing_counts = Counter(row["timing_quality"] for row in rows)
    heat_done = sum(1 for row in rows if row["heat_exists"])
    dino_done = sum(1 for row in rows if row["dinov2_points"] is not None)
    bad = [
        row for row in rows
        if row["timing_quality"] not in {"good", "review"}
        or not row["heat_exists"]
        or row.get("contact_heat_max", 0.0) <= 0.0
    ]

    print(f"processed_root: {cfg.paths.processed_root}")
    print(f"episodes: {len(rows)}")
    print(f"timing_quality: {dict(sorted(timing_counts.items()))}")
    print(f"dinov2_done: {dino_done}/{len(rows)}")
    print(f"contact_heatmap_done: {heat_done}/{len(rows)}")
    print(f"bad_or_incomplete: {len(bad)}")

    if rows:
        hand_bbox_counts = [row["hand_bbox"] for row in rows]
        hand_mask_counts = [row["hand_mask"] for row in rows]
        print(f"hand_bbox frames min/median/max: {min(hand_bbox_counts)}/{int(np.median(hand_bbox_counts))}/{max(hand_bbox_counts)}")
        print(f"hand_mask frames min/median/max: {min(hand_mask_counts)}/{int(np.median(hand_mask_counts))}/{max(hand_mask_counts)}")

    if args.show_bad and bad:
        print("bad_or_incomplete episodes:")
        for row in bad:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
