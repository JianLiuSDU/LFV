#!/usr/bin/env python3
"""Audit a cached Stage-2 dataset for shape and instance-split leakage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.cache_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    missing = [
        str(item["episode_id"])
        for item in records
        if not str(item.get("object_instance_id", "")).strip()
    ]
    by_split = {}
    groups = split_manifest.get("groups", split_manifest.get("splits", {}))
    for split, ids in groups.items():
        by_split[split] = {
            "episodes": list(ids),
            "count": len(ids),
            "instances": sorted(
                {
                    str(next(
                        (row.get("object_instance_id", "") for row in records
                         if row.get("episode_id") == episode),
                        ""
                    )).strip()
                    for episode in ids
                    if str(next(
                        (row.get("object_instance_id", "") for row in records
                         if row.get("episode_id") == episode),
                        ""
                    )).strip()
                }
            ),
        }
    nonempty_instances = {
        str(row.get("object_instance_id", "")).strip()
        for row in records
        if str(row.get("object_instance_id", "")).strip()
    }
    split_instance_sets = {
        split: set(info["instances"]) for split, info in by_split.items()
    }
    leakage = {}
    names = list(split_instance_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sorted(split_instance_sets[left] & split_instance_sets[right])
            leakage[f"{left}__{right}"] = overlap
    report = {
        "cache_root": str(root),
        "manifest_num_points": manifest.get("num_points"),
        "manifest_dino_dim": manifest.get("dino", {}).get("feature_dim"),
        "record_count": len(records),
        "missing_object_instance_id_count": len(missing),
        "missing_object_instance_ids": missing,
        "nonempty_instance_count": len(nonempty_instances),
        "splits": by_split,
        "instance_leakage": leakage,
        "valid_object_split": bool(
            not missing and all(not values for values in leakage.values())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
