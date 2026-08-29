#!/usr/bin/env python3
"""Run the repository SAM2 box-mask helper in its dedicated environment.

The parent strict inference process deliberately does not import SAM2.  This
worker is launched with the Python executable from the user's SAM2 conda
environment and calls the existing ``lfv.pipeline.sam2_mask._load_predictor``
and box-prompt logic without changing its segmentation algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True, help="npz with cup and bowl boxes")
    parser.add_argument("--output", type=Path, required=True, help="npz for masks and scores")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--perception-config", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    sam2_root = args.sam2_root.expanduser().resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(sam2_root))

    from lfv.pipeline.sam2_mask import _load_predictor
    from lfv.utils.config import load_config

    cfg = load_config(args.perception_config.expanduser().resolve())
    cfg.sam2.model_cfg = str(args.model_cfg)
    cfg.sam2.checkpoint = str(args.checkpoint.expanduser().resolve())
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    predictor = _load_predictor(cfg, device)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    boxes = np.load(args.boxes, allow_pickle=False)
    predictor.set_image(rgb)
    masks: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    enabled = device.startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
        for name in ("cup", "bowl"):
            box = np.asarray(boxes[name], dtype=np.float32).reshape(1, 4)
            predicted, confidence, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )
            index = int(np.argmax(confidence))
            masks[name] = np.asarray(predicted[index], dtype=bool)
            scores[name] = float(confidence[index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, cup_mask=masks["cup"], bowl_mask=masks["bowl"])
    args.output.with_suffix(".json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scores": scores}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
