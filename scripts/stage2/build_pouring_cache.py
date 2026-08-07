#!/usr/bin/env python3
"""Build the offline 256-point XYZ-DINO Stage 2 cache."""

from __future__ import annotations

import argparse
import json

from lfv.datasets.functional_motion.cache_builder import build_dataset_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instance-mapping")
    parser.add_argument("--strict-instance-split", action="store_true")
    args = parser.parse_args()
    manifest = build_dataset_cache(
        args.source_root,
        args.cache_root,
        args.weights,
        device=args.device,
        num_points=args.num_points,
        overwrite=args.overwrite,
        limit=args.limit,
        instance_mapping_path=args.instance_mapping,
        allow_episode_id_as_instance=not args.strict_instance_split,
    )
    print(json.dumps(manifest["audit"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
