#!/usr/bin/env python3
"""Export a trained Stage 2 relevance field and aligned descriptors as memory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lfv.models.functional_motion_generation import load_stage2_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True, help="cached Stage 2 episode .npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    with np.load(args.artifact, allow_pickle=False) as data:
        batch = {
            key: torch.from_numpy(np.asarray(data[key], dtype=np.float32))[None].to(device)
            for key in ("manipulated_points", "manipulated_dino", "reference_points", "reference_dino")
        }
    model, config, _ = load_stage2_checkpoint(args.checkpoint, device=device, use_ema=True)
    with torch.inference_mode():
        encoding = model.encode(batch, return_debug=True)
    if encoding.manipulated_motion_field is None or encoding.reference_motion_field is None:
        raise RuntimeError("Checkpoint does not expose motion fields; use a motion_field_mode=joint/independent checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        manipulated_points=batch["manipulated_points"][0].cpu().numpy(),
        manipulated_dino=batch["manipulated_dino"][0].cpu().numpy(),
        manipulated_motion_field=encoding.manipulated_motion_field[0].cpu().numpy(),
        reference_points=batch["reference_points"][0].cpu().numpy(),
        reference_dino=batch["reference_dino"][0].cpu().numpy(),
        reference_motion_field=encoding.reference_motion_field[0].cpu().numpy(),
    )
    print({"output": str(args.output.resolve()), "num_points": int(batch["manipulated_points"].shape[1]), "dino_dim": int(batch["manipulated_dino"].shape[-1]), "model": config["model"]["name"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
