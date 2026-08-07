#!/usr/bin/env python3
"""Audit all source episodes without modifying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfv.datasets.functional_motion.audit import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-points", type=int, default=256)
    args = parser.parse_args()
    report = audit_dataset(args.source_root, num_points=args.num_points)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "episodes": report["episode_count"],
                "accepted": report["accepted_count"],
                "rejected": report["rejected_count"],
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if report["accepted_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
