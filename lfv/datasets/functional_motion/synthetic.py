"""Deterministic synthetic Stage 2 samples for overfit tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from lfv.geometry import identity_pose9d


class SyntheticFunctionalMotionDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 32,
        num_points: int = 256,
        dino_dim: int = 32,
        horizon: int = 64,
        seed: int = 7,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.manipulated = torch.randn(
            num_samples, num_points, 3, generator=generator
        ) * 0.04
        offsets = torch.randn(num_samples, 3, generator=generator) * 0.03
        offsets[:, 2] += 0.25
        self.reference = (
            torch.randn(num_samples, num_points, 3, generator=generator) * 0.05
            + offsets[:, None]
        )
        projection = torch.randn(3, dino_dim, generator=generator)
        self.manipulated_dino = torch.nn.functional.normalize(
            self.manipulated @ projection, dim=-1
        )
        self.reference_dino = torch.nn.functional.normalize(
            self.reference @ projection, dim=-1
        )
        goal = identity_pose9d(num_samples)
        goal[:, :3] = offsets * torch.tensor([0.8, 0.8, 0.5])
        progress = torch.linspace(0.0, 1.0, horizon)[None, :, None]
        trajectory = identity_pose9d(num_samples, horizon)
        trajectory[:, :, :3] = goal[:, None, :3] * progress
        self.goal = goal
        self.trajectory = trajectory

    def __len__(self) -> int:
        return len(self.goal)

    def __getitem__(self, index: int):
        return {
            "manipulated_points": self.manipulated[index],
            "manipulated_dino": self.manipulated_dino[index],
            "reference_points": self.reference[index],
            "reference_dino": self.reference_dino[index],
            "goal_pose9d": self.goal[index],
            "trajectory_pose9d": self.trajectory[index],
            "scene_origin": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
            "episode_id": f"synthetic_{index:03d}",
            "object_instance_id": f"synthetic_{index:03d}",
        }
