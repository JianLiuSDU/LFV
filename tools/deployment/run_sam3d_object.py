#!/usr/bin/env python3
"""Small SAM3D-Objects bridge; runs only in the configured SAM3D environment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo / "notebook"))
    from inference import Inference  # type: ignore
    inference = Inference(args.config, compile=False)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    output = inference(rgb, mask, seed=args.seed)
    mesh = output["mesh"]
    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0]
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32) if hasattr(mesh, "faces") else np.empty((0, 3), dtype=np.int32)
    np.savez_compressed(args.output, points_camera=vertices, mesh_vertices=vertices, mesh_faces=faces, confidence=np.array(1.0, dtype=np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
