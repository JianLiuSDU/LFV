"""Reproducible split manifests with explicit leakage policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def build_split_manifest(
    records: Iterable[dict],
    output_path: str | Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    allow_episode_id_as_instance: bool = False,
) -> dict:
    records = list(records)
    if not records:
        raise ValueError("Cannot split an empty record list")
    groups: dict[str, list[str]] = {}
    provisional = False
    for record in records:
        instance = str(record.get("object_instance_id", "")).strip()
        if not instance:
            if not allow_episode_id_as_instance:
                raise ValueError(
                    "object_instance_id is missing; pass the explicit baseline opt-in "
                    "allow_episode_id_as_instance=True or provide a mapping"
                )
            instance = str(record["episode_id"])
            provisional = True
        groups.setdefault(instance, []).append(str(record["episode_id"]))
    ordered = sorted(groups, key=lambda value: _key(value, seed))
    n = len(ordered)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    group_splits = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }
    episode_splits = {
        split: sorted(ep for group in values for ep in groups[group])
        for split, values in group_splits.items()
    }
    manifest = {
        "seed": seed,
        "ratios": list(ratios),
        "split_quality": "episode_split_baseline" if provisional else "instance_disjoint",
        "allow_episode_id_as_instance": bool(allow_episode_id_as_instance),
        "groups": group_splits,
        "episodes": episode_splits,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
