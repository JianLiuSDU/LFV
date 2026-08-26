"""Small dependency-free exponential moving average."""

from __future__ import annotations

import copy

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.named_parameters()
            if value.requires_grad and torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = dict(model.named_parameters())
        for key, shadow in self.shadow.items():
            shadow.lerp_(state[key].detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        loaded = state["shadow"]
        self.shadow = {
            key: value.detach().clone().to(
                device=self.shadow[key].device,
                dtype=self.shadow[key].dtype,
            )
            for key, value in loaded.items()
            if key in self.shadow
        }

    def copy_to(self, model: nn.Module) -> None:
        state = dict(model.named_parameters())
        for key, value in self.shadow.items():
            if key in state:
                state[key].copy_(value)

    def averaged_model(self, model: nn.Module) -> nn.Module:
        averaged = copy.deepcopy(model)
        self.copy_to(averaged)
        return averaged
