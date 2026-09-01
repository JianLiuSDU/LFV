#!/usr/bin/env python3
"""Pull a target RGB-D point set onto V7 source-canonical support.

The correspondence NPZ is produced by an external DINO/FGW/cycle matcher and
must contain ``manipulated_correspondence`` and ``reference_correspondence``.
No arithmetic online/prior field fusion is performed here: the output gate is
the canonical source field multiplied by correspondence confidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfv.models.functional_motion_generation.canonical_alignment import (
    CanonicalFieldMemory,
    canonical_field_gate,
    pull_target_to_source,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    memory = CanonicalFieldMemory.load(args.memory)
    with np.load(args.target, allow_pickle=False) as target, np.load(
        args.correspondence, allow_pickle=False
    ) as corr:
        output: dict[str, np.ndarray] = {}
        for role in ("manipulated", "reference"):
            points_key = role + "_points"
            dino_key = role + "_dino"
            corr_key = role + "_correspondence"
            mask_key = role + "_mask"
            for key in (points_key, dino_key):
                if key not in target.files:
                    raise KeyError(f"target input is missing {key}")
            if corr_key not in corr.files:
                raise KeyError(f"correspondence input is missing {corr_key}")
            points = np.asarray(target[points_key], dtype=np.float32)
            dino = np.asarray(target[dino_key], dtype=np.float32)
            mask = (
                np.asarray(target[mask_key], dtype=bool)
                if mask_key in target.files
                else None
            )
            aligned_points, aligned_dino, confidence = pull_target_to_source(
                np.asarray(corr[corr_key], dtype=np.float32),
                points,
                dino,
                target_mask=mask,
            )
            field = getattr(memory, role + "_field_mean")
            gate = canonical_field_gate(field, confidence)
            output[role + "_points"] = aligned_points
            output[role + "_dino"] = aligned_dino
            # A row with an infinitesimal correspondence mass is not a valid
            # observation.  Keep the canonical support shape, but expose a
            # conservative visibility mask for the V7 encoder.
            output[role + "_mask"] = (confidence > 1e-6).astype(np.float32)
            output[role + "_source_canonical_field"] = field.astype(np.float32)
            output[role + "_correspondence_confidence"] = confidence
            output[role + "_field_gate"] = gate
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
    print({"output": str(args.output.resolve()), "source_canonical": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
