#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = [
    "rgb",
    "depth",
    "bbox/affordance_bbox.npy",
    "sam_mask/affordance_mask.npy",
    "sample_points/sampled_2d_uniform.npy",
    "target_bbox/target_bbox.npy",
    "target_sam_mask/target_mask.npy",
    "target_sample_points/target_sampled_2d_uniform.npy",
    "point_tracking/tapip3d_result.npz",
    "se3_trajectory/dp_action_trajectory.npz",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_root")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.processed_root)
    episodes = sorted(root.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1]))
    missing_total = 0
    for ep in episodes[: args.limit]:
        missing = [rel for rel in REQUIRED if not (ep / rel).exists()]
        if missing:
            missing_total += 1
            print(f"{ep.name}: missing {', '.join(missing)}")
        else:
            print(f"{ep.name}: ok")
    print(f"checked={min(len(episodes), args.limit)} missing_episodes={missing_total}")


if __name__ == "__main__":
    main()
