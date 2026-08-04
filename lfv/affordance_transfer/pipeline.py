from __future__ import annotations

from typing import Any

import numpy as np

from lfv.features.base import DenseFeatureExtractor

from .confidence import compute_transfer_confidence
from .preprocessing import (
    foreground_grid,
    map_grid_to_original,
    normalize_inside_mask,
    prepare_image,
    reduce_to_feature_grid,
)
from .schema import SourceContactExample, TargetObservation, TransferResult
from .soft_affcorrs import soft_heatmap_affcorrs


def _get(mapping: dict[str, Any], key: str, default: Any) -> Any:
    return mapping[key] if key in mapping else default


def _heat_location(heatmap: np.ndarray) -> dict[str, Any]:
    heatmap = np.asarray(heatmap, dtype=np.float64)
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    mass = float(heatmap.sum())
    yy, xx = np.indices(heatmap.shape)
    if mass > 1e-12:
        centroid_uv = [
            float(np.sum(xx * heatmap) / mass),
            float(np.sum(yy * heatmap) / mass),
        ]
    else:
        centroid_uv = [float("nan"), float("nan")]
    return {
        "peak_uv": [int(peak_x), int(peak_y)],
        "centroid_uv": centroid_uv,
        "mass": mass,
    }


class SoftHeatmapAffCorrsPipeline:
    """Pure 2D source-to-target continuous affordance transfer."""

    def __init__(self, extractor: DenseFeatureExtractor, config: dict[str, Any]) -> None:
        self.extractor = extractor
        self.config = config

    def transfer(
        self, source: SourceContactExample, target: TargetObservation
    ) -> TransferResult:
        prep_cfg = self.config.get("preprocessing", {})
        matching_cfg = self.config.get("matching", {})
        confidence_cfg = self.config.get("confidence", {})
        input_size = int(_get(prep_cfg, "input_size", 518))
        common = {
            "input_size": input_size,
            "patch_size": self.extractor.patch_size,
            "bbox_margin": float(_get(prep_cfg, "bbox_margin", 0.15)),
        }
        source_prepared = prepare_image(
            source.rgb, source.mask, heatmap=source.heatmap, **common
        )
        target_prepared = prepare_image(target.rgb, target.mask, **common)
        source_grid_features = self.extractor.extract(source_prepared.rgb)
        target_grid_features = self.extractor.extract(target_prepared.rgb)
        if source_grid_features.shape[:2] != target_grid_features.shape[:2]:
            raise RuntimeError("Source and target DINO grids have different shapes.")
        grid_hw = source_grid_features.shape[:2]
        occupancy_threshold = float(_get(prep_cfg, "mask_occupancy_threshold", 0.35))
        source_fg = foreground_grid(
            source_prepared, grid_hw, occupancy_threshold=occupancy_threshold
        )
        target_fg = foreground_grid(
            target_prepared, grid_hw, occupancy_threshold=occupancy_threshold
        )
        source_heat_grid = reduce_to_feature_grid(source_prepared.heatmap, grid_hw)
        source_heat_grid *= source_fg
        if float(source_heat_grid.max()) > 0:
            source_heat_grid /= float(source_heat_grid.max())

        source_features = source_grid_features[source_fg]
        target_features = target_grid_features[target_fg]
        source_heat = source_heat_grid[source_fg]
        matching = soft_heatmap_affcorrs(
            source_features,
            source_heat,
            target_features,
            source_clusters=int(_get(matching_cfg, "source_clusters", 6)),
            target_clusters=int(_get(matching_cfg, "target_clusters", 64)),
            positive_threshold=float(_get(matching_cfg, "positive_threshold", 0.2)),
            forward_temperature=float(_get(matching_cfg, "forward_temperature", 0.1)),
            backward_temperature=float(
                _get(matching_cfg, "backward_temperature", 0.05)
            ),
            seed=int(_get(matching_cfg, "seed", 0)),
            n_init=int(_get(matching_cfg, "n_init", 4)),
            max_iter=int(_get(matching_cfg, "max_iter", 100)),
        )

        target_labels = matching.target_clustering.labels
        target_patch_scores = matching.target_cluster_scores[target_labels]
        forward_patch_scores = matching.forward_votes[target_labels]
        backward_patch_scores = matching.backward_scores[target_labels]
        cycle_grid = np.zeros(grid_hw, dtype=np.float32)
        forward_grid = np.zeros_like(cycle_grid)
        backward_grid = np.zeros_like(cycle_grid)
        target_cluster_grid = np.full(grid_hw, -1, dtype=np.int16)
        cycle_grid[target_fg] = target_patch_scores
        forward_grid[target_fg] = forward_patch_scores
        backward_grid[target_fg] = backward_patch_scores
        target_cluster_grid[target_fg] = target_labels.astype(np.int16)

        source_cluster_grid = np.full(grid_hw, -1, dtype=np.int16)
        source_positive_grid = np.zeros(grid_hw, dtype=bool)
        positive_positions = np.flatnonzero(source_fg)[matching.source_positive_mask]
        source_cluster_grid.flat[positive_positions] = matching.source_clustering.labels
        source_positive_grid.flat[positive_positions] = True

        raw = map_grid_to_original(
            cycle_grid, target_prepared.transform, original_mask=target.mask
        )
        normalized, flat_target = normalize_inside_mask(raw, target.mask)
        confidence = compute_transfer_confidence(
            matching,
            target_patch_scores,
            minimum_retained_heat=float(
                _get(confidence_cfg, "minimum_retained_heat", 0.5)
            ),
            minimum_cycle_score=float(
                _get(confidence_cfg, "minimum_cycle_score", 0.05)
            ),
            minimum_peak_score=float(_get(confidence_cfg, "minimum_peak_score", 0.05)),
            maximum_entropy=float(_get(confidence_cfg, "maximum_entropy", 0.98)),
            minimum_global_score=float(
                _get(confidence_cfg, "minimum_global_score", 0.05)
            ),
        )
        rejection_reasons = list(confidence.rejection_reasons)
        if flat_target:
            rejection_reasons.append("flat_target_heatmap")

        diagnostics: dict[str, Any] = {
            "source_id": source.sample_id,
            "target_id": target.sample_id,
            "source_heat_location": _heat_location(source.heatmap),
            "target_heat_location": _heat_location(normalized),
            "source_transform": source_prepared.transform.to_dict(),
            "target_transform": target_prepared.transform.to_dict(),
            "feature_grid_hw": list(grid_hw),
            "feature_dim": int(source_grid_features.shape[-1]),
            "source_foreground_patch_count": int(source_fg.sum()),
            "source_positive_patch_count": int(matching.source_positive_mask.sum()),
            "target_foreground_patch_count": int(target_fg.sum()),
            "source_effective_clusters": int(matching.source_clustering.centers.shape[0]),
            "target_effective_clusters": int(matching.target_clustering.centers.shape[0]),
            "source_heat_grid": source_heat_grid,
            "source_positive_grid": source_positive_grid,
            "source_cluster_grid": source_cluster_grid,
            "target_cluster_grid": target_cluster_grid,
            "forward_vote_grid": forward_grid,
            "backward_score_grid": backward_grid,
            "cycle_score_grid": cycle_grid,
        }
        return TransferResult(
            target_heatmap=normalized,
            target_heatmap_raw=raw,
            confidence=confidence.values,
            accepted=not rejection_reasons,
            rejection_reasons=rejection_reasons,
            diagnostics=diagnostics,
        )
