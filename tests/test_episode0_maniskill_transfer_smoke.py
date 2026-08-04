from pathlib import Path

import cv2
import numpy as np

from lfv.affordance_transfer.adapters import load_target_snapshot
from lfv.affordance_transfer.pipeline import SoftHeatmapAffCorrsPipeline
from lfv.affordance_transfer.schema import SourceContactExample, TargetObservation


class _ColorPatchExtractor:
    patch_size = 14

    def extract(self, rgb):
        features = cv2.resize(rgb.astype(np.float32) / 255.0, (8, 8))
        yy, xx = np.mgrid[:8, :8].astype(np.float32)
        features = np.concatenate(
            [features, xx[..., None] / 8.0, yy[..., None] / 8.0], axis=-1
        )
        return features / np.maximum(
            np.linalg.norm(features, axis=-1, keepdims=True), 1e-6
        )


def test_pipeline_smoke_preserves_target_shape_and_mask():
    source_rgb = np.full((96, 128, 3), 20, dtype=np.uint8)
    target_rgb = source_rgb.copy()
    source_mask = np.zeros((96, 128), dtype=bool)
    target_mask = np.zeros_like(source_mask)
    source_mask[20:80, 25:105] = True
    target_mask[18:82, 20:110] = True
    source_rgb[source_mask] = (80, 80, 180)
    target_rgb[target_mask] = (80, 80, 180)
    source_rgb[35:55, 25:45] = (220, 30, 20)
    target_rgb[34:56, 20:43] = (220, 30, 20)
    heat = np.zeros_like(source_mask, dtype=np.float32)
    heat[35:55, 25:45] = 1.0
    source = SourceContactExample(source_rgb, source_mask, heat, "synthetic_source")
    target = TargetObservation(target_rgb, target_mask, "synthetic_target")
    config = {
        "preprocessing": {"input_size": 112, "bbox_margin": 0.1},
        "matching": {
            "source_clusters": 2,
            "target_clusters": 8,
            "positive_threshold": 0.2,
            "n_init": 2,
        },
        "confidence": {
            "minimum_cycle_score": 0.0,
            "minimum_peak_score": 0.0,
            "maximum_entropy": 1.0,
            "minimum_global_score": 0.0,
        },
    }
    result = SoftHeatmapAffCorrsPipeline(_ColorPatchExtractor(), config).transfer(
        source, target
    )
    assert result.target_heatmap.shape == target_mask.shape
    assert result.target_heatmap.dtype == np.float32
    assert np.all(result.target_heatmap[~target_mask] == 0)
    assert float(result.target_heatmap.max()) == 1.0
    peak_u, peak_v = result.diagnostics["target_heat_location"]["peak_uv"]
    assert target_mask[peak_v, peak_u]


def test_target_adapter_does_not_touch_depth_or_point_cloud_keys(tmp_path: Path):
    path = tmp_path / "snapshot.npz"
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 1
    np.savez(
        path,
        rgb=rgb,
        cup_mask=mask,
        depth_m=np.array([object()], dtype=object),
        full_points_camera=np.array([object()], dtype=object),
    )
    observation = load_target_snapshot(path)
    assert observation.rgb.shape == rgb.shape
    assert observation.mask.sum() == mask.sum()
