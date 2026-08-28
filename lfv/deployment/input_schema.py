from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _read_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"RGB image must be [H,W,3], got {image.shape} from {path}")
    return image[..., :3].astype(np.uint8, copy=False)


def _read_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=False)
        key = "depth_m" if "depth_m" in payload else payload.files[0]
        depth = payload[key]
    else:
        depth = np.asarray(Image.open(path))
    depth = np.asarray(depth).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth must be [H,W], got {depth.shape} from {path}")
    return depth.astype(np.float32) * float(scale)


def _read_intrinsic(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("intrinsic_cv", payload.get("matrix", payload.get("K")))
    if matrix is None and all(key in payload for key in ("fx", "fy", "cx", "cy")):
        matrix = [[payload["fx"], 0.0, payload["cx"]],
                  [0.0, payload["fy"], payload["cy"]],
                  [0.0, 0.0, 1.0]]
    if matrix is None:
        raise ValueError(f"{path} must contain intrinsic_cv/matrix/K or fx,fy,cx,cy")
    intrinsic = np.asarray(matrix, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Camera intrinsic must be [3,3], got {intrinsic.shape}")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    return intrinsic


def _read_optional_mask(root: Path, name: str) -> np.ndarray | None:
    path = _find_first(root, (f"{name}_mask.png", f"{name}_mask.npy", f"{name}.png"))
    if path is None:
        return None
    values = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.asarray(Image.open(path))
    values = np.asarray(values).squeeze()
    return values.astype(bool)


@dataclass(frozen=True)
class CameraInput:
    """One aligned RGB-D observation in the OpenCV camera frame."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic_cv: np.ndarray
    cup_mask: np.ndarray | None = None
    bowl_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m, dtype=np.float32).squeeze()
        intrinsic = np.asarray(self.intrinsic_cv, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb must be [H,W,3], got {rgb.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth shape {depth.shape} does not match RGB {rgb.shape[:2]}")
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic_cv must be [3,3], got {intrinsic.shape}")
        for name, mask in (("cup_mask", self.cup_mask), ("bowl_mask", self.bowl_mask)):
            if mask is not None and np.asarray(mask).squeeze().shape != depth.shape:
                raise ValueError(f"{name} must match depth shape {depth.shape}")
        object.__setattr__(self, "rgb", rgb.astype(np.uint8, copy=False))
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsic_cv", intrinsic)
        if self.cup_mask is not None:
            object.__setattr__(self, "cup_mask", np.asarray(self.cup_mask).squeeze().astype(bool))
        if self.bowl_mask is not None:
            object.__setattr__(self, "bowl_mask", np.asarray(self.bowl_mask).squeeze().astype(bool))

    def validate(self, *, minimum_depth_m: float = 1e-4, maximum_depth_m: float = 5.0) -> dict[str, Any]:
        finite = np.isfinite(self.depth_m)
        valid = finite & (self.depth_m >= minimum_depth_m) & (self.depth_m <= maximum_depth_m)
        report: dict[str, Any] = {
            "rgb_shape": list(self.rgb.shape),
            "depth_shape": list(self.depth_m.shape),
            "valid_depth_ratio": float(valid.mean()),
            "depth_min_m": float(self.depth_m[valid].min()) if np.any(valid) else None,
            "depth_max_m": float(self.depth_m[valid].max()) if np.any(valid) else None,
            "camera_convention": self.metadata.get("camera_convention", "opencv_camera"),
        }
        for name, mask in (("cup", self.cup_mask), ("bowl", self.bowl_mask)):
            if mask is not None:
                report[f"{name}_mask_pixels"] = int(mask.sum())
                report[f"{name}_valid_depth_ratio"] = float(valid[mask].mean()) if np.any(mask) else 0.0
        if report["valid_depth_ratio"] <= 0:
            raise ValueError("Input depth contains no valid metric values")
        return report


def load_camera_input(root: str | Path) -> CameraInput:
    """Load a single RGB-D observation from a configurable input directory."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rgb_path = root / manifest.get("rgb_path", "rgb.png")
    depth_path = root / manifest.get("depth_path", "depth.npy")
    intrinsic_path = root / manifest.get("intrinsics_path", "intrinsics.json")
    if not rgb_path.exists():
        found = _find_first(root, ("rgb.png", "rgb.jpg", "color.png", "image.png"))
        if found is None:
            raise FileNotFoundError(f"Missing RGB input in {root}")
        rgb_path = found
    if not depth_path.exists():
        found = _find_first(root, ("depth.npy", "depth.npz", "depth.png", "depth.tiff"))
        if found is None:
            raise FileNotFoundError(f"Missing depth input in {root}")
        depth_path = found
    if not intrinsic_path.exists():
        found = _find_first(root, ("intrinsics.json", "camera.json", "camera_intrinsics.json"))
        if found is None:
            raise FileNotFoundError(f"Missing camera intrinsics JSON in {root}")
        intrinsic_path = found
    depth_scale = float(manifest.get("depth_scale", 1.0))
    rgb = _read_image(rgb_path)
    depth = _read_depth(depth_path, depth_scale)
    intrinsic = _read_intrinsic(intrinsic_path)
    return CameraInput(
        rgb=rgb,
        depth_m=depth,
        intrinsic_cv=intrinsic,
        cup_mask=_read_optional_mask(root, "cup"),
        bowl_mask=_read_optional_mask(root, "bowl"),
        metadata={"root": str(root), "manifest": manifest, "rgb_path": str(rgb_path),
                  "depth_path": str(depth_path), "intrinsics_path": str(intrinsic_path)},
    )
