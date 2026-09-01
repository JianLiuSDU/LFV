#!/usr/bin/env python3
"""Train Stage 2 from a YAML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from lfv.datasets.functional_motion import (
    FunctionalMotionDataset,
    SyntheticFunctionalMotionDataset,
    collate_functional_motion,
)
from lfv.models.functional_motion_generation import build_model
from lfv.models.functional_motion_generation.loading import model_kwargs
from lfv.training.functional_motion import FunctionalMotionTrainer, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["runtime"]["device"] = args.device
    requested = str(config["runtime"]["device"])
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not bool(config["runtime"].get("allow_cpu_fallback", False)):
            raise RuntimeError("CUDA requested but unavailable")
        config["runtime"]["device"] = "cpu"
    seed_everything(int(config["runtime"]["seed"]))
    data = config["data"]
    if data["type"] == "synthetic":
        train_dataset = SyntheticFunctionalMotionDataset(
            num_samples=int(data.get("num_samples", 32)),
            num_points=int(data.get("num_points", 64)),
            dino_dim=int(data["dino_dim"]),
            horizon=64,
            seed=int(config["runtime"]["seed"]),
        )
        val_dataset = train_dataset
        collate_fn = None
    else:
        train_dataset = FunctionalMotionDataset(
            data["cache_root"],
            "train",
            shuffle_points=bool(data.get("shuffle_points", True)),
            seed=int(config["runtime"]["seed"]),
            limit=data.get("train_limit"),
            consistency_group_fallback=data.get("consistency_group_fallback"),
        )
        if bool(data.get("overfit_same_as_train", False)):
            val_dataset = train_dataset
        else:
            val_dataset = FunctionalMotionDataset(
                data["cache_root"],
                "val",
                shuffle_points=False,
                seed=int(config["runtime"]["seed"]),
                limit=data.get("val_limit"),
                consistency_group_fallback=data.get("consistency_group_fallback"),
            )
        config["data"]["dino_dim"] = train_dataset.dino_dim
        collate_fn = collate_functional_motion
    model = build_model(
        config["model"]["name"],
        **model_kwargs(config, int(config["data"]["dino_dim"])),
    )
    output = Path(args.output_dir or config["paths"]["output_dir"]).expanduser()
    trainer = FunctionalMotionTrainer(
        model,
        train_dataset,
        val_dataset,
        config,
        output,
        collate_fn=collate_fn,
    )
    result = trainer.train(resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
