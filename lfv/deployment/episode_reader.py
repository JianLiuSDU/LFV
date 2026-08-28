from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .input_schema import CameraInput


@dataclass(frozen=True)
class EpisodeFrame:
    frame_index: int
    camera: CameraInput


def _open_array(path: Path):
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Episode RGB-D reading requires zarr") from exc
    return zarr.open(str(path), mode="r")


def read_episode_frame(episode_dir: str | Path, frame_index: int = 0) -> EpisodeFrame:
    root = Path(episode_dir).expanduser().resolve()
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episode meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rgb_store, depth_store = root / "rgb", root / "depth"
    rgb_array, depth_array = _open_array(rgb_store), _open_array(depth_store)
    if frame_index < 0 or frame_index >= rgb_array.shape[0]:
        raise IndexError(f"frame_index={frame_index} outside [0,{rgb_array.shape[0]})")
    rgb = np.asarray(rgb_array[frame_index], dtype=np.uint8)
    depth_raw = np.asarray(depth_array[frame_index])
    depth_scale = float(meta.get("depth_scale", 1.0))
    if depth_scale > 0.01 and np.issubdtype(depth_raw.dtype, np.integer):
        depth_scale = 0.001
    depth_m = depth_raw.astype(np.float32) * depth_scale
    intr = meta.get("color_intrinsics", meta.get("intrinsics", {}))
    intrinsic = np.array([[intr["fx"], 0.0, intr.get("ppx", intr.get("cx"))], [0.0, intr["fy"], intr.get("ppy", intr.get("cy"))], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera = CameraInput(rgb, depth_m, intrinsic, metadata={"episode_dir": str(root), "frame_index": int(frame_index), "camera_convention": "opencv_camera", "depth_scale": depth_scale, "meta": meta})
    return EpisodeFrame(int(frame_index), camera)


def read_episode_sequence(episode_dir: str | Path, frame_indices: list[int] | np.ndarray | None = None) -> list[EpisodeFrame]:
    root = Path(episode_dir).expanduser().resolve()
    length = int(_open_array(root / "rgb").shape[0])
    indices = list(range(length)) if frame_indices is None else [int(i) for i in frame_indices]
    return [read_episode_frame(root, index) for index in indices]
