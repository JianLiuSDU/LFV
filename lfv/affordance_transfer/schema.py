from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_rgb(rgb: np.ndarray, name: str) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {rgb.shape}.")
    if rgb.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {rgb.dtype}.")
    return rgb


def _validate_mask(mask: np.ndarray, hw: tuple[int, int], name: str) -> np.ndarray:
    mask = np.asarray(mask).squeeze()
    if mask.shape != hw:
        raise ValueError(f"{name} must have shape {hw}, got {mask.shape}.")
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError(f"{name} is empty.")
    return mask


@dataclass(frozen=True)
class SourceContactExample:
    rgb: np.ndarray
    mask: np.ndarray
    heatmap: np.ndarray
    sample_id: str

    def __post_init__(self) -> None:
        rgb = _validate_rgb(self.rgb, "source_rgb")
        mask = _validate_mask(self.mask, rgb.shape[:2], "source_mask")
        heatmap = np.asarray(self.heatmap, dtype=np.float32).squeeze()
        if heatmap.shape != rgb.shape[:2]:
            raise ValueError(
                f"source_heatmap must have shape {rgb.shape[:2]}, got {heatmap.shape}."
            )
        if not np.all(np.isfinite(heatmap)):
            raise ValueError("source_heatmap contains non-finite values.")
        heatmap = np.clip(heatmap, 0.0, 1.0) * mask
        if float(heatmap.max()) <= 0:
            raise ValueError("source_heatmap has no positive value inside source_mask.")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "heatmap", heatmap)


@dataclass(frozen=True)
class TargetObservation:
    rgb: np.ndarray
    mask: np.ndarray
    sample_id: str

    def __post_init__(self) -> None:
        rgb = _validate_rgb(self.rgb, "target_rgb")
        mask = _validate_mask(self.mask, rgb.shape[:2], "target_mask")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class RGBDPart:
    """Aligned depth and camera model for one visible functional part.

    Coordinates follow the OpenCV camera convention: +x right, +y down and
    +z forward.  ``part_mask`` identifies the *whole* functional part used by
    FGW, rather than only the high-contact pixels.
    """

    depth_m: np.ndarray
    intrinsic_cv: np.ndarray
    part_mask: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"depth_m must have shape [H,W], got {depth.shape}.")
        intrinsic = np.asarray(self.intrinsic_cv, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            raise ValueError(
                f"intrinsic_cv must have shape [3,3], got {intrinsic.shape}."
            )
        if not np.all(np.isfinite(intrinsic)):
            raise ValueError("intrinsic_cv contains non-finite values.")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("Camera focal lengths must be positive.")
        part_mask = _validate_mask(self.part_mask, depth.shape, "part_mask")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsic_cv", intrinsic)
        object.__setattr__(self, "part_mask", part_mask)


@dataclass(frozen=True)
class CropTransform:
    original_hw: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    resized_hw: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    input_hw: tuple[int, int]
    patch_size: int

    def original_to_input(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        x0, y0, x1, y1 = self.crop_xyxy
        resized_h, resized_w = self.resized_hw
        left, top, _, _ = self.padding_ltrb
        scale_x = resized_w / max(x1 - x0, 1)
        scale_y = resized_h / max(y1 - y0, 1)
        result = xy.copy()
        result[..., 0] = (xy[..., 0] - x0) * scale_x + left
        result[..., 1] = (xy[..., 1] - y0) * scale_y + top
        return result

    def input_to_original(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        x0, y0, x1, y1 = self.crop_xyxy
        resized_h, resized_w = self.resized_hw
        left, top, _, _ = self.padding_ltrb
        scale_x = resized_w / max(x1 - x0, 1)
        scale_y = resized_h / max(y1 - y0, 1)
        result = xy.copy()
        result[..., 0] = (xy[..., 0] - left) / scale_x + x0
        result[..., 1] = (xy[..., 1] - top) / scale_y + y0
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_hw": list(self.original_hw),
            "crop_xyxy": list(self.crop_xyxy),
            "resized_hw": list(self.resized_hw),
            "padding_ltrb": list(self.padding_ltrb),
            "input_hw": list(self.input_hw),
            "patch_size": self.patch_size,
        }


@dataclass(frozen=True)
class PreparedImage:
    rgb: np.ndarray
    mask: np.ndarray
    content_mask: np.ndarray
    transform: CropTransform
    heatmap: np.ndarray | None = None


@dataclass
class TransferResult:
    target_heatmap: np.ndarray
    target_heatmap_raw: np.ndarray
    confidence: dict[str, float]
    accepted: bool
    rejection_reasons: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)
