"""Cached functional-motion PyTorch Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import validate_functional_motion_sample


class FunctionalMotionDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        shuffle_points: bool = True,
        seed: int = 42,
        limit: int | None = None,
        consistency_group_fallback: str | None = None,
    ) -> None:
        self.root = Path(cache_root)
        self.split = split
        self.shuffle_points = bool(shuffle_points)
        self.seed = int(seed)
        self.epoch = 0
        self.consistency_group_fallback = (
            str(consistency_group_fallback).strip()
            if consistency_group_fallback is not None
            else ""
        )
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        split_manifest = json.loads(
            (self.root / "split_manifest.json").read_text(encoding="utf-8")
        )
        allowed = set(split_manifest["episodes"][split])
        self.records = [
            record for record in manifest["records"] if record["episode_id"] in allowed
        ]
        if limit is not None:
            self.records = self.records[: int(limit)]
        if not self.records:
            raise ValueError(f"No records for split={split} in {self.root}")
        self.dino_dim = int(self.records[0]["dino_dim"])
        self.split_quality = str(manifest["split_quality"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with np.load(record["artifact"], allow_pickle=False) as data:
            raw_instance_id = str(data["object_instance_id"])
            object_instance_id = raw_instance_id or str(data["episode_id"])
            consistency_group = raw_instance_id or self.consistency_group_fallback or str(data["episode_id"])
            sample: dict[str, Any] = {
                "manipulated_points": data["manipulated_points"].astype(np.float32),
                "manipulated_dino": data["manipulated_dino"].astype(np.float32),
                "reference_points": data["reference_points"].astype(np.float32),
                "reference_dino": data["reference_dino"].astype(np.float32),
                "goal_pose9d": data["goal_pose9d"].astype(np.float32),
                "trajectory_pose9d": data["trajectory_pose9d"].astype(np.float32),
                "scene_origin": data["scene_origin"].astype(np.float32),
                "scene_scale": np.asarray(data["scene_scale"], dtype=np.float32),
                "episode_id": str(data["episode_id"]),
                "object_instance_id": object_instance_id,
                "field_consistency_group": consistency_group,
            }
        if self.shuffle_points:
            rng = np.random.default_rng(
                self.seed + 1_000_003 * self.epoch + 97 * int(index)
            )
            for prefix in ("manipulated", "reference"):
                permutation = rng.permutation(sample[f"{prefix}_points"].shape[0])
                sample[f"{prefix}_points"] = sample[f"{prefix}_points"][permutation]
                sample[f"{prefix}_dino"] = sample[f"{prefix}_dino"][permutation]
        validate_functional_motion_sample(sample)
        for key in (
            "manipulated_points",
            "manipulated_dino",
            "reference_points",
            "reference_dino",
            "goal_pose9d",
            "trajectory_pose9d",
            "scene_origin",
            "scene_scale",
        ):
            sample[key] = torch.from_numpy(np.ascontiguousarray(sample[key])).float()
        return sample


def collate_functional_motion(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        output[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return output
