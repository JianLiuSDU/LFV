"""Training-split translation statistics for Pose9D."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class Pose9DNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("translation_mean", torch.zeros(3))
        self.register_buffer("translation_std", torch.ones(3))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_tensors(self, poses: list[torch.Tensor]) -> "Pose9DNormalizer":
        if not poses:
            raise ValueError("Cannot fit Pose9DNormalizer on no poses")
        translations = torch.cat(
            [torch.as_tensor(pose, dtype=torch.float32).reshape(-1, 9)[:, :3] for pose in poses],
            dim=0,
        )
        self.translation_mean.copy_(translations.mean(dim=0))
        self.translation_std.copy_(
            translations.std(dim=0, unbiased=False).clamp_min(self.eps)
        )
        self.fitted.fill_(True)
        return self

    def normalize(self, pose: torch.Tensor) -> torch.Tensor:
        translation = (pose[..., :3] - self.translation_mean) / self.translation_std
        return torch.cat((translation, pose[..., 3:9]), dim=-1)

    def denormalize(self, pose: torch.Tensor) -> torch.Tensor:
        translation = pose[..., :3] * self.translation_std + self.translation_mean
        return torch.cat((translation, pose[..., 3:9]), dim=-1)

    def to_dict(self) -> dict:
        return {
            "translation_mean": self.translation_mean.detach().cpu().tolist(),
            "translation_std": self.translation_std.detach().cpu().tolist(),
            "fitted": bool(self.fitted.item()),
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, state: dict) -> "Pose9DNormalizer":
        normalizer = cls()
        normalizer.translation_mean.copy_(
            torch.as_tensor(state["translation_mean"], dtype=torch.float32)
        )
        normalizer.translation_std.copy_(
            torch.as_tensor(state["translation_std"], dtype=torch.float32)
        )
        normalizer.fitted.fill_(bool(state.get("fitted", True)))
        return normalizer
