"""Compatibility helpers for processed LFV episode formats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_episode_calibration(episode: str | Path) -> tuple[float, np.ndarray, str]:
    """Return depth scale, OpenCV intrinsic and the calibration source.

    Current processed episodes store both values in ``meta.json``.  The older
    82-episode pouring set predates that file: its Zarr depth is already in
    metres and TAPIP3D saved the camera intrinsics alongside its tracks.  This
    fallback keeps the source dataset immutable while exposing one cache
    contract to the new Stage 2 models.
    """

    root = Path(episode)
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta["depth_intrinsics_original"]
        intrinsic = np.asarray(
            [
                [raw["fx"], 0.0, raw["ppx"]],
                [0.0, raw["fy"], raw["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return float(meta["depth_scale"]), intrinsic, "meta.json"

    tracking_path = root / "point_tracking/tapip3d_result.npz"
    if not tracking_path.exists():
        raise FileNotFoundError(
            f"Episode {root} has neither meta.json nor TAPIP3D calibration"
        )
    with np.load(tracking_path, allow_pickle=False) as tracking:
        intrinsics = np.asarray(tracking["intrinsics"], dtype=np.float32)
        intrinsic = intrinsics[0] if intrinsics.ndim == 3 else intrinsics
        # Legacy pouring Zarr depths are already metric.  Newer tracking files
        # may carry the raw sensor conversion explicitly.
        depth_scale = (
            float(np.asarray(tracking["depth_scale"]).reshape(()))
            if "depth_scale" in tracking.files
            else 1.0
        )
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"Invalid legacy intrinsic in {tracking_path}: {intrinsic}")
    return depth_scale, intrinsic.astype(np.float32), "tapip3d_result.npz"
