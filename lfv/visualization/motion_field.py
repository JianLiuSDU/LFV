"""Small, deterministic visualizations for online/prior/fused motion fields."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _overlay(rgb: np.ndarray, pixels: np.ndarray, field: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    values = np.asarray(field, dtype=np.float32).reshape(-1)
    values = values / max(float(values.max()), 1e-8)
    colors = plt.get_cmap("turbo")(values)[:, :3] * 255.0
    for (u, v), color, value in zip(np.asarray(pixels, dtype=np.int32), colors, values):
        if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
            radius = 2 + int(round(3.0 * float(value)))
            cv2.circle(image, (int(u), int(v)), radius, tuple(int(x) for x in color[::-1]), -1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_motion_field_comparison(
    rgb: np.ndarray,
    manipulated_pixels: np.ndarray,
    reference_pixels: np.ndarray,
    manipulated_online: np.ndarray,
    reference_online: np.ndarray,
    manipulated_prior: np.ndarray | None,
    reference_prior: np.ndarray | None,
    manipulated_fused: np.ndarray,
    reference_fused: np.ndarray,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("online manipulated", manipulated_pixels, manipulated_online),
        ("memory prior manipulated", manipulated_pixels, manipulated_prior),
        ("fused manipulated", manipulated_pixels, manipulated_fused),
        ("online reference", reference_pixels, reference_online),
        ("memory prior reference", reference_pixels, reference_prior),
        ("fused reference", reference_pixels, reference_fused),
    ]
    for axis, (title, pixels, field) in zip(axes.flat, panels):
        axis.imshow(rgb if field is None else _overlay(rgb, pixels, field))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Stage 2 Motion Functional Field: online / memory prior / fused")
    figure.savefig(output, dpi=160)
    plt.close(figure)
