#!/usr/bin/env python3
"""Build a V7 source-canonical Motion Field memory.

Each ``--record`` is required to contain the episode field and an explicit
episode-to-canonical mapping.  The script intentionally refuses the old
episode-id-only fallback: averaging fields in different camera/sample orders
would create a false canonical support.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfv.models.functional_motion_generation.canonical_alignment import (
    CanonicalFieldMemory,
    aggregate_canonical_fields,
)


def _read(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} is missing {key}; available={data.files}")
        return np.asarray(data[key], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--canonical", type=Path, required=True,
        help="NPZ with manipulated/reference canonical points and DINO",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs="*", type=float)
    args = parser.parse_args()
    if args.weights and len(args.weights) != len(args.records):
        raise ValueError("--weights must have one value per --records path")
    weights = args.weights or [1.0] * len(args.records)
    with np.load(args.canonical, allow_pickle=False) as canonical:
        required = (
            "manipulated_points", "manipulated_dino",
            "reference_points", "reference_dino",
        )
        missing = [key for key in required if key not in canonical.files]
        if missing:
            raise KeyError(f"canonical support is missing {missing}")
        canonical_values = {
            key: np.asarray(canonical[key], dtype=np.float32) for key in required
        }
    fields: dict[str, list[np.ndarray]] = {"manipulated": [], "reference": []}
    mappings: dict[str, list[np.ndarray]] = {"manipulated": [], "reference": []}
    for record in args.records:
        for role in fields:
            # Keep the names explicit so a contact memory or target heatmap
            # cannot be silently mistaken for a V7 motion field.
            fields[role].append(_read(record, f"{role}_motion_field"))
            mappings[role].append(_read(record, f"{role}_episode_to_canonical"))
    result: dict[str, np.ndarray] = {}
    for role in fields:
        mean, variance, confidence = aggregate_canonical_fields(
            fields[role], mappings[role], sample_weights=weights
        )
        result[role + "_field_mean"] = mean
        result[role + "_field_var"] = variance
        result[role + "_confidence"] = confidence
    memory = CanonicalFieldMemory(
        manipulated_points=canonical_values["manipulated_points"],
        manipulated_dino=canonical_values["manipulated_dino"],
        manipulated_field_mean=result["manipulated_field_mean"],
        manipulated_field_var=result["manipulated_field_var"],
        manipulated_confidence=result["manipulated_confidence"],
        reference_points=canonical_values["reference_points"],
        reference_dino=canonical_values["reference_dino"],
        reference_field_mean=result["reference_field_mean"],
        reference_field_var=result["reference_field_var"],
        reference_confidence=result["reference_confidence"],
    )
    memory.save(args.output)
    print({"output": str(args.output.resolve()), "records": len(args.records)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

