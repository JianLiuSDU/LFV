#!/usr/bin/env python3
"""Sweep target-region K-Means counts for a fixed Soft Heatmap AffCorrs case."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.affordance_transfer.adapters import load_source_episode, load_target_snapshot
from lfv.affordance_transfer.io import save_transfer_result
from lfv.affordance_transfer.pipeline import SoftHeatmapAffCorrsPipeline
from lfv.features import DinoV2DenseExtractor
from lfv.utils.config import load_config
from lfv.visualization.affordance_transfer import (
    render_transfer_source_target_2x2,
    render_transfer_summary,
)


def _overlay(rgb: np.ndarray, heat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    color = plt.get_cmap("turbo")(heat)[..., :3]
    base = np.asarray(rgb, dtype=np.float32) / 255.0
    alpha = (0.72 * heat + 0.08) * np.asarray(mask, dtype=np.float32)
    return np.clip(base * (1.0 - alpha[..., None]) + color * alpha[..., None], 0, 1)


def _mask_metrics(heat: np.ndarray, mask: np.ndarray) -> dict[str, float | list]:
    heat = np.asarray(heat, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.indices(mask.shape)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    one_third = (x1 - x0 + 1) / 3.0
    middle = mask & (xx >= x0 + one_third) & (xx < x0 + 2.0 * one_third)
    mass = float(heat[mask].sum())
    peak_y, peak_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    if mass > 1e-12:
        centroid = [
            float((xx[mask] * heat[mask]).sum() / mass),
            float((yy[mask] * heat[mask]).sum() / mass),
        ]
        probability = heat[mask] / mass
        positive = probability > 0
        pixel_entropy = float(
            -(probability[positive] * np.log(probability[positive])).sum()
            / np.log(len(probability))
        )
    else:
        centroid = [float("nan"), float("nan")]
        pixel_entropy = float("nan")
    return {
        "peak_uv": [int(peak_x), int(peak_y)],
        "centroid_uv": centroid,
        "central_third_mass_fraction": float(heat[middle].sum() / max(mass, 1e-12)),
        "coverage_gt_0.2": float(np.count_nonzero((heat > 0.2) & mask) / mask.sum()),
        "coverage_gt_0.5": float(np.count_nonzero((heat > 0.5) & mask) / mask.sum()),
        "pixel_entropy": pixel_entropy,
        "mass_outside_mask": float(heat[~mask].sum()),
    }


def _render_comparison(source, target, runs: list[dict], output: Path) -> None:
    panels = [
        (
            _overlay(source.rgb, source.heatmap, source.mask),
            "Source handle heat",
        )
    ]
    for run in runs:
        metrics = run["metrics"]
        panels.append(
            (
                _overlay(target.rgb, run["result"].target_heatmap, target.mask),
                (
                    f"target K={run['target_clusters']} | conf={run['result'].confidence['global']:.3f}\n"
                    f"middle={metrics['central_third_mass_fraction']:.3f} | "
                    f">0.5={metrics['coverage_gt_0.5']:.3f}"
                ),
            )
        )
    columns = min(4, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, panel in zip(axes.flat, panels):
        image, title = panel
        axis.imshow(image)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    fig.suptitle(
        "Handle-only Soft Heatmap AffCorrs: target K-Means cluster sweep",
        fontsize=16,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--target-clusters",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 96, 128, 160],
    )
    parser.add_argument("--device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    source_cfg = cfg.source
    target_cfg = cfg.target
    source = load_source_episode(
        source_cfg.episode_dir,
        frame_index=int(source_cfg.frame_index),
        mask_path=str(source_cfg.mask_path),
        heatmap_path=str(source_cfg.heatmap_path),
        heatmap_key=str(source_cfg.heatmap_key),
    )
    target = load_target_snapshot(
        target_cfg.snapshot_path,
        rgb_key=str(target_cfg.rgb_key),
        mask_key=str(target_cfg.mask_key),
    )
    device = str(args.device or cfg.runtime.device)
    extractor = DinoV2DenseExtractor(
        model_name=str(cfg.features.model_name),
        weights_path=str(cfg.features.weights_path),
        device=device,
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for cluster_count in args.target_clusters:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.matching.target_clusters = int(cluster_count)
        run_dir = output_root / f"target_clusters_{cluster_count:03d}"
        result = SoftHeatmapAffCorrsPipeline(extractor, run_cfg).transfer(source, target)
        paths = save_transfer_result(result, run_dir, config=run_cfg)
        paths["visualization"] = render_transfer_summary(
            source, target, result, run_dir / "transfer_summary.png"
        )
        paths["source_target_2x2"] = render_transfer_source_target_2x2(
            source, target, result, run_dir / "transfer_source_target_2x2.png"
        )
        runs.append(
            {
                "target_clusters": int(cluster_count),
                "result": result,
                "metrics": _mask_metrics(result.target_heatmap, target.mask),
                "paths": paths,
            }
        )
        print(
            json.dumps(
                {
                    "target_clusters": cluster_count,
                    "accepted": result.accepted,
                    "global_confidence": result.confidence["global"],
                    **runs[-1]["metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    comparison = output_root / "cluster_sweep_comparison.png"
    _render_comparison(source, target, runs, comparison)
    serializable = []
    for run in runs:
        serializable.append(
            {
                "target_clusters": run["target_clusters"],
                "accepted": run["result"].accepted,
                "rejection_reasons": run["result"].rejection_reasons,
                "confidence": run["result"].confidence,
                "metrics": run["metrics"],
                "effective_target_clusters": int(
                    run["result"].diagnostics["target_effective_clusters"]
                ),
                "outputs": {key: str(value) for key, value in run["paths"].items()},
            }
        )
    report = {
        "base_config": str(Path(args.config).expanduser().resolve()),
        "scope": "target cluster count only; no depth, point cloud, or GraspNet",
        "target_mask": str(target_cfg.mask_key),
        "source_clusters": int(cfg.matching.source_clusters),
        "runs": serializable,
        "comparison": str(comparison),
    }
    report_path = output_root / "cluster_sweep_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"comparison": str(comparison), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
