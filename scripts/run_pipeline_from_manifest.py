#!/usr/bin/env python3
"""Run unchanged LFV pipeline stages on episodes accepted by an audit manifest."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.utils.config import load_config
from scripts.run_pipeline import STAGES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    episodes = report.get("accepted_episodes", [])
    if not episodes:
        raise RuntimeError("Manifest contains no accepted episodes")
    cfg.runtime.episodes = episodes
    if args.overwrite:
        cfg.runtime.overwrite = True
    print(f"[manifest] accepted episodes: {len(episodes)}")
    for step in args.steps:
        if step not in STAGES:
            raise ValueError(f"Unknown stage {step!r}; known={sorted(STAGES)}")
        print(f"\n========== LFV manifest stage: {step} ==========", flush=True)
        importlib.import_module(STAGES[step]).run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
