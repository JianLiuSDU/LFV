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
from lfv.utils.config import get_nested, load_config


def _episode_name(value: str) -> str:
    return value if value.startswith("episode_") else f"episode_{value}"


def _load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_episode(ep: pathlib.Path, cfg) -> dict:
    out_dir = ep / str(get_nested(cfg, "thumb_index_grasp.output_dir", "hamer_grasp_pseudo_label"))
    hamer_meta = _load_json(out_dir / "hamer_output" / "hamer_run_meta.json")
    grasp_meta = _load_json(out_dir / "grasp_pseudo_label_meta.json")

    result = {
        "episode": ep.name,
        "hamer_frames_with_keypoints": len(hamer_meta.get("frames_with_keypoints", [])) if hamer_meta else 0,
        "grasp_exists": (out_dir / "grasp_pseudo_label.npz").exists(),
        "quality": grasp_meta.get("quality") if grasp_meta else "missing",
        "confidence": grasp_meta.get("confidence") if grasp_meta else None,
        "width_m": grasp_meta.get("width_m") if grasp_meta else None,
        "valid_candidate_count": grasp_meta.get("valid_candidate_count") if grasp_meta else None,
        "selected_frame": grasp_meta.get("selected_frame") if grasp_meta else None,
        "mean_finger_surface_dist_m": grasp_meta.get("mean_finger_surface_dist_m") if grasp_meta else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize hand pouring thumb-index grasp pseudo label outputs.")
    parser.add_argument("--config", default="configs/pipeline/hand_pouring.yaml")
    parser.add_argument("--episode", action="append", default=[], help="Episode id/name. Can be repeated.")
    parser.add_argument("--show-bad", action="store_true", help="Print episodes with missing/reject outputs.")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    selected = [_episode_name(ep) for ep in args.episode] if args.episode else None
    episodes = iter_processed_episodes(cfg.paths.processed_root, selected)

    rows = [summarize_episode(ep, cfg) for ep in episodes]
    quality_counts = Counter(row["quality"] for row in rows)
    grasp_done = sum(1 for row in rows if row["grasp_exists"])
    bad = [row for row in rows if row["quality"] not in {"good", "review"} or not row["grasp_exists"]]

    print(f"processed_root: {cfg.paths.processed_root}")
    print(f"episodes: {len(rows)}")
    print(f"quality: {dict(sorted(quality_counts.items()))}")
    print(f"grasp_done: {grasp_done}/{len(rows)}")
    print(f"bad_or_incomplete: {len(bad)}")

    good_rows = [row for row in rows if row["quality"] in {"good", "review"} and row["confidence"] is not None]
    if good_rows:
        conf = np.asarray([row["confidence"] for row in good_rows], dtype=np.float64)
        width = np.asarray([row["width_m"] for row in good_rows], dtype=np.float64)
        dist = np.asarray([row["mean_finger_surface_dist_m"] for row in good_rows], dtype=np.float64)
        print(f"confidence min/median/max: {conf.min():.3f}/{np.median(conf):.3f}/{conf.max():.3f}")
        print(f"width_m min/median/max: {width.min():.3f}/{np.median(width):.3f}/{width.max():.3f}")
        print(f"finger_surface_dist min/median/max: {dist.min():.3f}/{np.median(dist):.3f}/{dist.max():.3f}")

    if args.show_bad and bad:
        print("bad_or_incomplete episodes:")
        for row in bad:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
