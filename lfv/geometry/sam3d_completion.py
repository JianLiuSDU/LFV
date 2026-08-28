from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompletedObject:
    """Object geometry in the OpenCV camera frame."""

    visible_points_camera: np.ndarray
    complete_points_camera: np.ndarray
    visible_to_complete: np.ndarray | None
    camera_from_object: np.ndarray
    is_complete: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name, value in (("visible_points_camera", self.visible_points_camera), ("complete_points_camera", self.complete_points_camera)):
            array = np.asarray(value)
            if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
                raise ValueError(f"{name} must be non-empty [N,3], got {array.shape}")
        transform = np.asarray(self.camera_from_object)
        if transform.shape != (4, 4):
            raise ValueError("camera_from_object must be [4,4]")


def backproject_mask(mask: np.ndarray, depth_m: np.ndarray, intrinsic_cv: np.ndarray, *, max_depth_m: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > 1e-4) & (depth <= max_depth_m)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise ValueError("Object mask has no valid depth pixels")
    z = depth[v, u]
    k = np.asarray(intrinsic_cv, dtype=np.float32)
    points = np.stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z), axis=-1).astype(np.float32)
    return points, np.stack((u, v), axis=-1).astype(np.int32)


class VisibleOnlyCompletionBackend:
    """Deterministic fallback for pipeline smoke tests; never claims completion."""

    def __init__(self, max_depth_m: float = 5.0):
        self.max_depth_m = max_depth_m

    def complete(self, rgb: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, intrinsic_cv: np.ndarray, output_dir: str | Path) -> CompletedObject:
        points, pixels = backproject_mask(mask, depth_m, intrinsic_cv, max_depth_m=self.max_depth_m)
        result = CompletedObject(points, points.copy(), np.arange(len(points), dtype=np.int64), np.eye(4, dtype=np.float32), False, 0.0, {"backend": "visible_only", "pixel_count": int(len(pixels))})
        result.validate()
        return result


class NPZCompletionBackend:
    """Read a geometry-completion artifact generated offline on an A100."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def complete(self, rgb: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, intrinsic_cv: np.ndarray, output_dir: str | Path) -> CompletedObject:
        payload = np.load(self.path, allow_pickle=False)
        points_key = "complete_points_camera" if "complete_points_camera" in payload else "points_camera"
        if points_key not in payload:
            raise KeyError(f"Completion artifact requires {points_key}")
        visible, _ = backproject_mask(mask, depth_m, intrinsic_cv)
        transform = np.asarray(payload["camera_from_object"], dtype=np.float32) if "camera_from_object" in payload else np.eye(4, dtype=np.float32)
        result = CompletedObject(visible, np.asarray(payload[points_key], dtype=np.float32), None, transform, True, float(payload["confidence"]) if "confidence" in payload else 1.0, {"backend": "npz", "path": str(self.path)})
        result.validate()
        return result


class SAM3DSubprocessBackend:
    """Invoke SAM3D-Objects in its own environment and consume mesh vertices.

    The helper writes ``mesh_vertices.npy``.  Alignment is deliberately an
    explicit downstream step: SAM3D's canonical frame is not assumed to be a
    camera frame.
    """

    def __init__(self, python_executable: str, sam3d_repo: str, config_path: str, *, seed: int = 42):
        self.python_executable = str(python_executable)
        self.sam3d_repo = str(sam3d_repo)
        self.config_path = str(config_path)
        self.seed = int(seed)

    def complete(self, rgb: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, intrinsic_cv: np.ndarray, output_dir: str | Path) -> CompletedObject:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        rgb_path, mask_path = output_dir / "sam3d_rgb.png", output_dir / "sam3d_mask.png"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(rgb_path)
        Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255)).save(mask_path)
        helper = Path(__file__).resolve().parents[2] / "tools" / "deployment" / "run_sam3d_object.py"
        command = [self.python_executable, str(helper), "--repo", self.sam3d_repo, "--config", self.config_path, "--rgb", str(rgb_path), "--mask", str(mask_path), "--output", str(output_dir / "sam3d_mesh.npz"), "--seed", str(self.seed)]
        subprocess.run(command, check=True)
        payload = np.load(output_dir / "sam3d_mesh.npz", allow_pickle=False)
        canonical = np.asarray(payload["mesh_vertices"] if "mesh_vertices" in payload else payload["points_camera"], dtype=np.float32)
        visible, _ = backproject_mask(mask, depth_m, intrinsic_cv)
        from .registration import rigid_icp_to_visible
        transform, aligned, rms = rigid_icp_to_visible(canonical, visible)
        confidence = float(np.exp(-rms / 0.01))
        result = CompletedObject(visible, aligned, None, transform, True, confidence, {"backend": "sam3d", "canonical_vertices": int(len(canonical)), "registration_rms_m": rms, "registration_warning": "reject if RMS is too large"})
        result.validate()
        return result
