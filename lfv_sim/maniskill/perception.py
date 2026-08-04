from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .tensor_utils import squeeze_env_dim


@dataclass
class CameraObservation:
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray | None
    intrinsic_cv: np.ndarray
    extrinsic_cv: np.ndarray | None


def _first(mapping: dict, names: list[str]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def extract_camera_observation(obs: dict, camera_uid: str) -> CameraObservation:
    obs = squeeze_env_dim(obs)
    sensor = obs["sensor_data"][camera_uid]
    params = obs["sensor_param"][camera_uid]
    rgb = _first(sensor, ["rgb", "Color", "color"])
    depth = _first(sensor, ["depth", "Depth", "position"])
    segmentation = _first(sensor, ["segmentation", "Segmentation"])
    intrinsic = _first(params, ["intrinsic_cv", "intrinsic", "K"])
    extrinsic = _first(params, ["extrinsic_cv", "extrinsic"])
    if rgb is None or depth is None or intrinsic is None:
        raise KeyError(
            f"Incomplete camera observation for {camera_uid}: "
            f"sensor={sorted(sensor)}, params={sorted(params)}"
        )
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return CameraObservation(
        rgb=rgb,
        depth=np.asarray(depth),
        segmentation=None if segmentation is None else np.asarray(segmentation),
        intrinsic_cv=np.asarray(intrinsic, dtype=np.float32),
        extrinsic_cv=None if extrinsic is None else np.asarray(extrinsic, dtype=np.float32),
    )


def segmentation_id_map(env) -> dict[int, object]:
    return dict(getattr(getattr(env, "unwrapped", env), "segmentation_id_map", {}) or {})


def find_segmentation_ids(env, query: str) -> list[int]:
    query = query.lower()
    return sorted(
        {
            int(segmentation_id)
            for segmentation_id, actor in segmentation_id_map(env).items()
            if query in str(getattr(actor, "name", actor)).lower()
        }
    )


def object_mask(camera: CameraObservation, segmentation_ids: Iterable[int]) -> np.ndarray:
    if camera.segmentation is None:
        raise RuntimeError("ManiSkill observation does not contain segmentation")
    segmentation = camera.segmentation
    if segmentation.ndim == 3:
        segmentation = segmentation[..., 0]
    return np.isin(
        segmentation.astype(np.int64),
        np.asarray(list(segmentation_ids), dtype=np.int64),
    )
