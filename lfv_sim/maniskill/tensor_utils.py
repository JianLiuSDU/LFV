from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_numpy(item) for item in value)
    return value


def squeeze_env_dim(value: Any) -> Any:
    value = to_numpy(value)
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, dict):
        return {key: squeeze_env_dim(item) for key, item in value.items()}
    return value
