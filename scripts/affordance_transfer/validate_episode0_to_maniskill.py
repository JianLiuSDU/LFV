#!/usr/bin/env python3
"""Fixed quick-iteration validation case for episode_0 -> ManiSkill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfv.affordance_transfer.app import run_transfer
from lfv.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/affordance_transfer/episode0_to_maniskill.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    run = run_transfer(
        load_config(args.config),
        output_dir_override=args.output_dir,
        device_override=args.device,
    )
    result = run["result"]
    summary = {
        "accepted": result.accepted,
        "confidence": result.confidence,
        "rejection_reasons": result.rejection_reasons,
        "visualization": str(run["paths"]["visualization"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not result.accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
