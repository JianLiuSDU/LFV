#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.utils.config import load_config


STAGES = {
    "prepare": "lfv.pipeline.prepare",
    "dino": "lfv.pipeline.dino_bbox",
    "sam2": "lfv.pipeline.sam2_mask",
    "sample": "lfv.pipeline.sample_points",
    "track": "lfv.pipeline.tracking",
    "se3": "lfv.pipeline.se3_trajectory",
    "hand": "lfv.pipeline.hand_segmentation",
    "hand_bbox": "lfv.pipeline.hand_bbox",
    "hand_mask": "lfv.pipeline.hand_mask",
    "timing": "lfv.pipeline.contact_timing",
    "dinov2": "lfv.pipeline.dinov2_features",
    "contact_heatmap": "lfv.pipeline.contact_heatmap",
    "contact": "lfv.pipeline.contact_field",
    "hamer": "lfv.pipeline.hamer_hand_pose",
    "thumb_index_grasp": "lfv.pipeline.thumb_index_grasp_label",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LFV data processing pipeline stages.")
    parser.add_argument("--config", default="configs/pipeline/picknplace.yaml")
    parser.add_argument("--steps", nargs="+", default=["prepare", "dino", "sam2", "sample", "se3"])
    parser.add_argument("--episodes", nargs="*", default=None, help="Optional episode ids or names.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    if args.episodes is not None:
        cfg.runtime.episodes = args.episodes
    if args.overwrite:
        cfg.runtime.overwrite = True

    for step in args.steps:
        if step not in STAGES:
            raise ValueError(f"Unknown step {step!r}. Available: {sorted(STAGES)}")
        print(f"\n========== LFV stage: {step} ==========")
        module = importlib.import_module(STAGES[step])
        module.run(cfg)


if __name__ == "__main__":
    main()
