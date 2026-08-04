from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lfv.features import DinoV2DenseExtractor
from lfv.visualization.affordance_transfer import (
    render_affcorrs_fgw_comparison,
    render_transfer_source_target_2x2,
    render_transfer_summary,
)

from .adapters import (
    load_source_episode,
    load_source_rgbd_part,
    load_target_rgbd_part,
    load_target_snapshot,
)
from .fgw_contact_transfer import AffCorrsFGWContactTransferPipeline
from .io import save_transfer_result
from .pipeline import SoftHeatmapAffCorrsPipeline


def run_transfer(
    config: dict[str, Any],
    *,
    output_dir_override: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_cfg = config["source"]
    target_cfg = config["target"]
    feature_cfg = config["features"]
    runtime_cfg = config.get("runtime", {})
    source = load_source_episode(
        source_cfg["episode_dir"],
        frame_index=int(source_cfg["frame_index"]),
        mask_path=str(source_cfg.get("mask_path", "sam_mask/affordance_mask.npy")),
        heatmap_path=str(
            source_cfg.get("heatmap_path", "contact_heatmap/contact_heatmap.npz")
        ),
        heatmap_key=str(source_cfg.get("heatmap_key", "heatmap_2d")),
    )
    target = load_target_snapshot(
        target_cfg["snapshot_path"],
        rgb_key=str(target_cfg.get("rgb_key", "rgb")),
        mask_key=str(target_cfg.get("mask_key", "cup_mask")),
    )
    device = str(device_override or runtime_cfg.get("device", "cuda"))
    extractor = DinoV2DenseExtractor(
        model_name=str(feature_cfg.get("model_name", "vit_small_patch14_dinov2")),
        weights_path=feature_cfg["weights_path"],
        device=device,
    )
    method = str(config.get("method", "soft_heatmap_affcorrs"))
    if method == "soft_heatmap_affcorrs":
        pipeline = SoftHeatmapAffCorrsPipeline(extractor, config)
        result = pipeline.transfer(source, target)
    elif method == "affcorrs_fgw":
        source_rgbd = load_source_rgbd_part(
            source_cfg["episode_dir"],
            frame_index=int(source_cfg["frame_index"]),
            part_mask_path=str(
                source_cfg.get("part_mask_path", source_cfg["mask_path"])
            ),
            metadata_path=str(source_cfg.get("metadata_path", "meta.json")),
            intrinsics_key=str(source_cfg.get("intrinsics_key", "color_intrinsics")),
        )
        target_rgbd = load_target_rgbd_part(
            target_cfg["snapshot_path"],
            depth_key=str(target_cfg.get("depth_key", "depth_m")),
            intrinsics_key=str(target_cfg.get("intrinsics_key", "intrinsic_cv")),
            part_mask_key=str(
                target_cfg.get("part_mask_key", target_cfg["mask_key"])
            ),
        )
        pipeline = AffCorrsFGWContactTransferPipeline(extractor, config)
        result = pipeline.transfer(source, target, source_rgbd, target_rgbd)
    else:
        raise ValueError(
            f"Unknown affordance-transfer method {method!r}; expected "
            "'soft_heatmap_affcorrs' or 'affcorrs_fgw'."
        )
    output_dir = Path(
        output_dir_override or config["output"]["directory"]
    ).expanduser()
    paths = save_transfer_result(result, output_dir, config=config)
    if method == "affcorrs_fgw":
        paths["visualization"] = render_affcorrs_fgw_comparison(
            source, target, result, output_dir / "transfer_summary.png"
        )
        paths["affcorrs_fgw_comparison"] = paths["visualization"]
    else:
        paths["visualization"] = render_transfer_summary(
            source, target, result, output_dir / "transfer_summary.png"
        )
    paths["source_target_2x2"] = render_transfer_source_target_2x2(
        source, target, result, output_dir / "transfer_source_target_2x2.png"
    )
    return {
        "source": source,
        "target": target,
        "result": result,
        "paths": paths,
        "device": str(extractor.device),
    }
