from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lfv.affordance_transfer.schema import (
    SourceContactExample,
    TargetObservation,
    TransferResult,
)


def _overlay(rgb: np.ndarray, heat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    colors = plt.get_cmap("turbo")(heat)[..., :3]
    base = rgb.astype(np.float32) / 255.0
    alpha = (0.72 * heat + 0.08) * np.asarray(mask, dtype=np.float32)
    return np.clip(base * (1.0 - alpha[..., None]) + colors * alpha[..., None], 0, 1)


def _cluster_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    output = np.zeros((*labels.shape, 3), dtype=np.float32)
    valid = labels >= 0
    if np.any(valid):
        maximum = max(int(labels[valid].max()) + 1, 2)
        output[valid] = plt.get_cmap("tab20")(
            (labels[valid] % 20) / min(maximum - 1, 19)
        )[..., :3]
    return output


def render_transfer_summary(
    source: SourceContactExample,
    target: TargetObservation,
    result: TransferResult,
    output_path: str | Path,
) -> Path:
    """Save the fixed quick-iteration diagnostic figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = result.diagnostics

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].imshow(source.rgb)
    axes[0, 0].contour(source.mask, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[0, 0].set_title(f"Source RGB + object mask\n{source.sample_id}")

    axes[0, 1].imshow(_overlay(source.rgb, source.heatmap, source.mask))
    axes[0, 1].set_title("Source continuous contact heat")

    prototype_view = _cluster_rgb(diagnostics["source_cluster_grid"])
    positive = diagnostics["source_positive_grid"]
    prototype_view[~positive] *= 0.2
    axes[0, 2].imshow(prototype_view, interpolation="nearest")
    axes[0, 2].set_title(
        "Source heat-positive prototypes\n"
        f"{diagnostics['source_positive_patch_count']} patches / "
        f"{diagnostics['source_effective_clusters']} prototypes"
    )

    axes[1, 0].imshow(target.rgb)
    axes[1, 0].contour(target.mask, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[1, 0].set_title(f"Target RGB + object mask\n{target.sample_id}")

    axes[1, 1].imshow(_overlay(target.rgb, result.target_heatmap, target.mask))
    status = "ACCEPT" if result.accepted else "REJECT"
    peak_uv = diagnostics["target_heat_location"]["peak_uv"]
    axes[1, 1].set_title(
        f"Transferred continuous heat — {status}\n"
        f"confidence={result.confidence['global']:.3f}, peak=({peak_uv[0]}, {peak_uv[1]})"
    )

    forward = diagnostics["forward_vote_grid"]
    backward = diagnostics["backward_score_grid"]
    cycle = diagnostics["cycle_score_grid"]
    panels = []
    for values in (forward, backward, cycle):
        vmax = float(np.max(values))
        normalized = values / vmax if vmax > 0 else values
        panels.append(plt.get_cmap("magma")(normalized)[..., :3])
    separator = np.ones((panels[0].shape[0], 2, 3), dtype=np.float32)
    combined = np.concatenate(
        [panels[0], separator, panels[1], separator, panels[2]], axis=1
    )
    axes[1, 2].imshow(combined, interpolation="nearest")
    axes[1, 2].set_title(
        "Forward V | Backward Q | Product H\n"
        f"cycle={result.confidence['cycle']:.3f}, "
        f"entropy={result.confidence['entropy']:.3f}"
    )

    for axis in axes.flat:
        axis.axis("off")
    if result.rejection_reasons:
        fig.suptitle("Rejected: " + ", ".join(result.rejection_reasons), fontsize=11)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_transfer_source_target_2x2(
    source: SourceContactExample,
    target: TargetObservation,
    result: TransferResult,
    output_path: str | Path,
) -> Path:
    """Save the task-neutral four-panel view used for rapid comparison."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status = "ACCEPT" if result.accepted else "REJECT"
    panels = (
        (source.rgb, f"Source RGB\n{source.sample_id}"),
        (_overlay(source.rgb, source.heatmap, source.mask), "Source contact heat"),
        (target.rgb, f"Target simulation RGB\n{target.sample_id}"),
        (
            _overlay(target.rgb, result.target_heatmap, target.mask),
            f"Transferred heat — {status}\nconfidence={result.confidence['global']:.3f}",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (image, title) in zip(axes.flat, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=13)
        axis.axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_affcorrs_fgw_comparison(
    source: SourceContactExample,
    target: TargetObservation,
    result: TransferResult,
    output_path: str | Path,
) -> Path:
    """Render the fixed AffCorrs-vs-FGW quick-iteration diagnostic."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = result.diagnostics
    baseline = np.asarray(diagnostics["affcorrs_target_heatmap"], dtype=np.float32)
    fgw_raw = np.asarray(diagnostics["target_heatmap_fgw_raw"], dtype=np.float32)

    def display_normalized(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        output = np.zeros_like(values, dtype=np.float32)
        maximum = float(np.max(values[mask], initial=0.0))
        if maximum > 1e-8:
            output[mask] = values[mask] / maximum
        return output

    fgw_display = display_normalized(fgw_raw, target.mask)
    final_display = display_normalized(result.target_heatmap, target.mask)
    source_nodes = np.asarray(diagnostics["source_node_pixels_uv"])
    target_nodes = np.asarray(diagnostics["target_node_pixels_uv"])
    source_node_heat = np.asarray(diagnostics["source_node_heat"])
    target_node_heat = np.asarray(diagnostics["target_node_heat"])

    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    axes[0, 0].imshow(source.rgb)
    axes[0, 0].contour(source.mask, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[0, 0].set_title(f"Source RGB + FGW part mask\n{source.sample_id}")
    axes[0, 1].imshow(_overlay(source.rgb, source.heatmap, source.mask))
    axes[0, 1].set_title("Source continuous Contact Field")
    axes[0, 2].imshow(source.rgb)
    scatter = axes[0, 2].scatter(
        source_nodes[:, 0],
        source_nodes[:, 1],
        c=source_node_heat,
        cmap="turbo",
        vmin=0,
        vmax=1,
        s=12,
    )
    axes[0, 2].set_title(
        f"Source FGW nodes ({len(source_nodes)})\nheat retained on whole part"
    )
    fig.colorbar(scatter, ax=axes[0, 2], fraction=0.046)
    axes[0, 3].imshow(np.asarray(diagnostics["semantic_cost"]), cmap="viridis")
    axes[0, 3].set_title(
        "Cross-instance DINO semantic cost\n"
        f"FGW alpha={diagnostics['fgw_alpha']:.2f}"
    )

    axes[1, 0].imshow(target.rgb)
    axes[1, 0].contour(target.mask, levels=[0.5], colors=["lime"], linewidths=1.1)
    axes[1, 0].set_title(f"Target RGB + FGW part mask\n{target.sample_id}")
    axes[1, 1].imshow(_overlay(target.rgb, baseline, target.mask))
    axes[1, 1].set_title(
        "AffCorrs only (K=64)\nsemantic region, may spread across the part"
    )
    axes[1, 2].imshow(_overlay(target.rgb, fgw_display, target.mask))
    axes[1, 2].scatter(
        target_nodes[:, 0],
        target_nodes[:, 1],
        c=target_node_heat,
        cmap="turbo",
        vmin=0,
        vmax=1,
        s=7,
        alpha=0.55,
    )
    axes[1, 2].set_title(
        f"FGW transported field (display normalized)\n{diagnostics['fgw_solver']}"
    )
    axes[1, 3].imshow(_overlay(target.rgb, final_display, target.mask))
    axes[1, 3].set_title(
        "Final Contact Field used downstream\n"
        f"raw max={float(result.target_heatmap.max()):.3f}, "
        f"confidence={result.confidence['global']:.3f}"
    )
    for axis in axes.flat:
        axis.axis("off")
    if result.rejection_reasons:
        fig.suptitle("Rejected: " + ", ".join(result.rejection_reasons), fontsize=11)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
