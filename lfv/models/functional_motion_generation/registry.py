"""Stage 2 model registry."""

from __future__ import annotations

from typing import Callable

from torch import nn


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def decorator(factory):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Duplicate Stage 2 model: {name}")
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown Stage 2 model {name}; available={sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
