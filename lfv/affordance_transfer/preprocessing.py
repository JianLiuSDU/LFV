from __future__ import annotations

import cv2
import numpy as np

from .schema import CropTransform, PreparedImage


def _expanded_mask_bbox(mask: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot crop an empty mask.")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(1, int(round((x1 - x0) * margin)))
    pad_y = max(1, int(round((y1 - y0) * margin)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def prepare_image(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    heatmap: np.ndarray | None = None,
    input_size: int = 518,
    patch_size: int = 14,
    bbox_margin: float = 0.15,
) -> PreparedImage:
    if input_size % patch_size:
        raise ValueError("input_size must be divisible by patch_size.")
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and mask spatial shapes differ.")

    x0, y0, x1, y1 = _expanded_mask_bbox(mask, bbox_margin)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)
    crop_h, crop_w = crop_mask.shape
    scale = min(input_size / crop_w, input_size / crop_h)
    resized_w = max(1, min(input_size, int(round(crop_w * scale))))
    resized_h = max(1, min(input_size, int(round(crop_h * scale))))
    left = (input_size - resized_w) // 2
    top = (input_size - resized_h) // 2
    right = input_size - resized_w - left
    bottom = input_size - resized_h - top

    resized_rgb = cv2.resize(crop_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        crop_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    fill = np.median(crop_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
    output_rgb = np.empty((input_size, input_size, 3), dtype=np.uint8)
    output_rgb[...] = fill
    output_rgb[top : top + resized_h, left : left + resized_w] = resized_rgb
    output_mask = np.zeros((input_size, input_size), dtype=bool)
    output_mask[top : top + resized_h, left : left + resized_w] = resized_mask
    content_mask = np.zeros_like(output_mask)
    content_mask[top : top + resized_h, left : left + resized_w] = True

    output_heat = None
    if heatmap is not None:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if heatmap.shape != mask.shape:
            raise ValueError("Heatmap and mask spatial shapes differ.")
        crop_heat = heatmap[y0:y1, x0:x1]
        resized_heat = cv2.resize(
            crop_heat, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        output_heat = np.zeros((input_size, input_size), dtype=np.float32)
        output_heat[top : top + resized_h, left : left + resized_w] = resized_heat
        output_heat *= output_mask

    transform = CropTransform(
        original_hw=mask.shape,
        crop_xyxy=(x0, y0, x1, y1),
        resized_hw=(resized_h, resized_w),
        padding_ltrb=(left, top, right, bottom),
        input_hw=(input_size, input_size),
        patch_size=patch_size,
    )
    return PreparedImage(
        rgb=output_rgb,
        mask=output_mask,
        heatmap=output_heat,
        content_mask=content_mask,
        transform=transform,
    )


def reduce_to_feature_grid(
    array: np.ndarray,
    grid_hw: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    grid_h, grid_w = grid_hw
    return cv2.resize(
        np.asarray(array, dtype=np.float32),
        (grid_w, grid_h),
        interpolation=interpolation,
    ).astype(np.float32, copy=False)


def foreground_grid(
    prepared: PreparedImage,
    grid_hw: tuple[int, int],
    *,
    occupancy_threshold: float = 0.35,
) -> np.ndarray:
    occupancy = reduce_to_feature_grid(prepared.mask, grid_hw)
    valid = reduce_to_feature_grid(prepared.content_mask, grid_hw)
    return (occupancy >= occupancy_threshold) & (valid >= 0.99)


def map_grid_to_original(
    grid: np.ndarray,
    transform: CropTransform,
    *,
    original_mask: np.ndarray | None = None,
) -> np.ndarray:
    input_h, input_w = transform.input_hw
    input_map = cv2.resize(
        np.asarray(grid, dtype=np.float32),
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    left, top, _, _ = transform.padding_ltrb
    resized_h, resized_w = transform.resized_hw
    content = input_map[top : top + resized_h, left : left + resized_w]
    x0, y0, x1, y1 = transform.crop_xyxy
    crop_map = cv2.resize(content, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    output = np.zeros(transform.original_hw, dtype=np.float32)
    output[y0:y1, x0:x1] = crop_map
    if original_mask is not None:
        output *= np.asarray(original_mask, dtype=bool)
    return output


def normalize_inside_mask(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    output = np.zeros_like(values)
    selected = values[mask]
    if selected.size == 0:
        return output, True
    low, high = float(selected.min()), float(selected.max())
    flat = high - low <= 1e-8
    if not flat:
        output[mask] = (selected - low) / (high - low)
    return output, flat
