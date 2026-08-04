#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfv.affordance_transfer.app import run_transfer
from lfv.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer one continuous source contact heatmap to a target RGB image."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/affordance_transfer/episode0_to_maniskill.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = run_transfer(
        cfg, output_dir_override=args.output_dir, device_override=args.device
    )
    result = run["result"]
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "rejection_reasons": result.rejection_reasons,
                "confidence": result.confidence,
                "device": run["device"],
                "outputs": {key: str(value) for key, value in run["paths"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
