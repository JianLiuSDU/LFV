from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small dict wrapper with attribute access for YAML configs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _wrap(data)


def get_nested(cfg: Config, dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
