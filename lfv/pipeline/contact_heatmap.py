from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial import cKDTree

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.pipeline.contact_field import (
    _as_bool_mask,
    _connected_components_high_heat,
    _load_se3_relative,
    _transform_anchor_points,
    _write_3d_visual,
    build_anchor_point_cloud,
)
from lfv.pipeline.tracking import load_episode_camera_params, project_to_2d
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def _cfg_get(cfg, dotted_key: str, default=None):
    cur = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_timing(ep_path: Path, cfg) -> dict:
    rel = str(_cfg_get(cfg, "contact_heatmap.timing_path", "contact_timing/contact_timing.json"))
    path = ep_path / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing contact timing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_anchor_object_mask(ep_path: Path, cfg) -> np.ndarray:
    rel = str(_cfg_get(cfg, "contact_heatmap.anchor_object_mask_path", "sam_mask/affordance_mask.npy"))
    path = ep_path / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing anchor object mask: {path}")
    return _as_bool_mask(np.load(path, allow_pickle=True))


def _load_object_mask_for_frame(ep_path: Path, cfg, frame: int, anchor_mask: np.ndarray) -> tuple[np.ndarray, str]:
    masks_dir = _cfg_get(cfg, "contact_heatmap.object_masks_dir")
    if masks_dir:
        pattern = str(_cfg_get(cfg, "contact_heatmap.object_mask_pattern", "frame_{frame:06d}.npy"))
        path = ep_path / str(masks_dir) / pattern.format(frame=frame)
        if path.exists():
            return _as_bool_mask(np.load(path, allow_pickle=True)), "per_frame"
    rel = str(_cfg_get(cfg, "contact_heatmap.object_mask_path", "sam_mask/affordance_mask.npy"))
    path = ep_path / rel
    if path.exists():
        return _as_bool_mask(np.load(path, allow_pickle=True)), "static_anchor"
    return anchor_mask, "anchor_fallback"


def _load_hand_mask_for_frame(ep_path: Path, cfg, frame: int) -> np.ndarray:
    masks_dir = ep_path / str(_cfg_get(cfg, "contact_heatmap.hand_masks_dir", "hand_mask"))
    pattern = str(_cfg_get(cfg, "contact_heatmap.hand_mask_pattern", "frame_{frame:06d}.npy"))
    path = masks_dir / pattern.format(frame=frame)
    if not path.exists():
        raise FileNotFoundError(f"Missing hand mask for contact frame {frame}: {path}")
    return _as_bool_mask(np.load(path, allow_pickle=True))


def _available_hand_frames(ep_path: Path, cfg) -> list[int]:
    masks_dir = ep_path / str(_cfg_get(cfg, "contact_heatmap.hand_masks_dir", "hand_mask"))
    frames = []
    for path in masks_dir.glob("frame_*.npy"):
        try:
            frames.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(frames)


def _select_contact_frames(ep_path: Path, cfg, timing: dict) -> list[int]:
    if _cfg_get(cfg, "contact_heatmap.frames") is not None:
        return [int(x) for x in _cfg_get(cfg, "contact_heatmap.frames")]
    contact_start = timing.get("contact_start")
    if contact_start is None:
        raise ValueError("contact_timing has no contact_start.")
    offsets = _cfg_get(cfg, "contact_heatmap.frame_offsets", [-3, 0, 3, 6])
    requested = [int(contact_start) + int(o) for o in offsets]
    available = _available_hand_frames(ep_path, cfg)
    if not available:
        raise FileNotFoundError(f"No hand masks found under {ep_path}")
    selected = []
    for frame in requested:
        nearest = min(available, key=lambda x: abs(x - frame))
        selected.append(int(nearest))
    num_frames = int(_cfg_get(cfg, "contact_heatmap.num_frames", 4))
    deduped = []
    for frame in selected:
        if frame not in deduped:
            deduped.append(frame)
    if len(deduped) < num_frames:
        center = int(contact_start)
        for frame in sorted(available, key=lambda x: abs(x - center)):
            if frame not in deduped:
                deduped.append(frame)
            if len(deduped) >= num_frames:
                break
    return sorted(deduped[:num_frames])


def _bbox_sigma(object_mask: np.ndarray, cfg) -> float:
    ys, xs = np.nonzero(object_mask)
    if len(xs) == 0:
        return float(_cfg_get(cfg, "contact_heatmap.sigma_min_px", 5.0))
    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    sigma = diag * float(_cfg_get(cfg, "contact_heatmap.sigma_bbox_fraction", 0.08))
    sigma = max(float(_cfg_get(cfg, "contact_heatmap.sigma_min_px", 5.0)), sigma)
    sigma = min(float(_cfg_get(cfg, "contact_heatmap.sigma_max_px", 18.0)), sigma)
    return float(sigma)


def _frame_pixel_evidence(hand_mask: np.ndarray, object_mask: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    dist = cv2.distanceTransform((~hand_mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    evidence = np.exp(-(dist ** 2) / (2.0 * sigma_px ** 2)).astype(np.float32)
    evidence *= object_mask.astype(np.float32)
    return evidence, dist


def _aggregate_topk(evidence_stack: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    if evidence_stack.ndim != 2:
        raise ValueError(f"Expected evidence stack [T,N], got {evidence_stack.shape}")
    if evidence_stack.shape[0] == 0:
        raise ValueError("No contact evidence frames to aggregate.")
    k = min(max(1, int(_cfg_get(cfg, "contact_heatmap.top_k", 2))), evidence_stack.shape[0])
    topk = np.sort(evidence_stack, axis=0)[-k:, :]
    agg = np.mean(topk, axis=0).astype(np.float32)
    active_thr = float(_cfg_get(cfg, "contact_heatmap.active_evidence_threshold", 0.25))
    active_count = np.sum(evidence_stack >= active_thr, axis=0).astype(np.int32)
    min_active = int(_cfg_get(cfg, "contact_heatmap.min_active_frames", 1))
    agg[active_count < min_active] = 0.0
    return agg, active_count


def _point_seed_components(points_uv: np.ndarray, seed_mask: np.ndarray, image_shape: tuple[int, int], cfg) -> tuple[np.ndarray, dict]:
    if not np.any(seed_mask):
        return seed_mask, {"num_components": 0, "kept_components": 0, "dropped_points": 0}
    h, w = image_shape
    binary = np.zeros((h, w), dtype=np.uint8)
    uv = points_uv[seed_mask]
    binary[np.clip(uv[:, 1], 0, h - 1), np.clip(uv[:, 0], 0, w - 1)] = 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return seed_mask, {"num_components": 0, "kept_components": 0, "dropped_points": 0}
    min_area = int(_cfg_get(cfg, "contact_heatmap.min_seed_component_area", 8))
    comp_ids = list(range(1, n))
    comp_ids = [cid for cid in comp_ids if int(stats[cid, cv2.CC_STAT_AREA]) >= min_area]
    if not comp_ids:
        comp_ids = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)]
    # Keep the largest component; this follows the "single grasp region first" rule.
    keep_id = max(comp_ids, key=lambda cid: int(stats[cid, cv2.CC_STAT_AREA]))
    keep = seed_mask.copy()
    point_labels = labels[np.clip(points_uv[:, 1], 0, h - 1), np.clip(points_uv[:, 0], 0, w - 1)]
    keep &= point_labels == keep_id
    return keep, {
        "num_components": int(n - 1),
        "kept_components": 1,
        "kept_component_area": int(stats[keep_id, cv2.CC_STAT_AREA]),
        "dropped_points": int(np.sum(seed_mask) - np.sum(keep)),
    }


def _fit_weighted_ellipse_heatmap(
    pixels_uv: np.ndarray,
    evidence: np.ndarray,
    object_mask: np.ndarray,
    seed_mask: np.ndarray,
    cfg,
) -> tuple[np.ndarray, dict]:
    h, w = object_mask.shape
    if int(np.sum(seed_mask)) < 3:
        return np.zeros((h, w), dtype=np.float32), {"status": "empty", "seed_count": int(np.sum(seed_mask))}
    q = pixels_uv[seed_mask].astype(np.float64)
    weights = np.maximum(evidence[seed_mask].astype(np.float64), 1e-6)
    w_sum = float(np.sum(weights))
    mu = np.sum(q * weights[:, None], axis=0) / max(w_sum, 1e-8)
    delta = q - mu[None, :]
    cov_raw = (delta * weights[:, None]).T @ delta / max(w_sum, 1e-8)
    cov = float(_cfg_get(cfg, "contact_heatmap.gaussian_scale_factor", 1.6)) * cov_raw
    cov += np.eye(2, dtype=np.float64) * float(_cfg_get(cfg, "contact_heatmap.gaussian_regularization", 9.0))
    vals, vecs = np.linalg.eigh(cov)

    ys, xs = np.nonzero(object_mask)
    bbox_diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)) if len(xs) else 50.0
    min_axis = float(_cfg_get(cfg, "contact_heatmap.min_axis_px", 5.0))
    max_axis = bbox_diag * float(_cfg_get(cfg, "contact_heatmap.max_axis_bbox_fraction", 0.45))
    vals = np.clip(vals, min_axis ** 2, max_axis ** 2)
    cov = vecs @ np.diag(vals) @ vecs.T
    inv_cov = np.linalg.inv(cov)

    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.stack([xx - mu[0], yy - mu[1]], axis=-1).astype(np.float64)
    maha = np.einsum("...i,ij,...j->...", grid, inv_cov, grid)
    heat = np.exp(-0.5 * maha).astype(np.float32)
    heat *= object_mask.astype(np.float32)
    if float(np.max(heat)) > 0:
        heat /= float(np.max(heat))

    order = np.argsort(vals)[::-1]
    major_vec = vecs[:, order[0]]
    angle = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return heat.astype(np.float32), {
        "status": "ok",
        "seed_count": int(np.sum(seed_mask)),
        "center_uv": mu.astype(float).tolist(),
        "covariance_raw": cov_raw.astype(float).tolist(),
        "covariance": cov.astype(float).tolist(),
        "axis_std_px": np.sqrt(vals[order]).astype(float).tolist(),
        "axis_angle_deg": angle,
    }


def _write_frame_evidence_visual(rgb: np.ndarray, object_mask: np.ndarray, hand_mask: np.ndarray, evidence: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.contour(object_mask, colors="yellow", linewidths=1.0)
    ax.imshow(hand_mask, cmap="Blues", alpha=hand_mask.astype(np.float32) * 0.35)
    ax.imshow(evidence, cmap="magma", alpha=np.clip(evidence, 0, 1) * 0.75, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _write_point_evidence_visual(rgb: np.ndarray, object_mask: np.ndarray, pixels_uv: np.ndarray, values: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.contour(object_mask, colors="yellow", linewidths=1.0)
    if len(pixels_uv):
        sc = ax.scatter(pixels_uv[:, 0], pixels_uv[:, 1], c=values, s=7, cmap="magma", vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, fraction=0.035)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _write_heatmap_visual(rgb: np.ndarray, heatmap: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(heatmap, cmap="magma", alpha=np.clip(heatmap, 0, 1) * 0.75, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def process_episode(ep_path: str | Path, cfg) -> bool:
    ep_path = Path(ep_path)
    out_dir = ep_path / str(_cfg_get(cfg, "contact_heatmap.output_dir", "contact_heatmap"))
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "contact_heatmap.npz"
    if out_path.exists() and not bool(_cfg_get(cfg, "runtime.overwrite", False)):
        print(f"[contact_heatmap] skip existing {ep_path.name}")
        return True

    timing = _load_timing(ep_path, cfg)
    anchor_frame = int(timing["anchor_frame"])
    frames = _select_contact_frames(ep_path, cfg, timing)

    rgb = zarr.open(str(ep_path / "rgb"), mode="r")
    depth_raw = zarr.open(str(ep_path / "depth"), mode="r")
    intrinsics, depth_scale, _meta = load_episode_camera_params(ep_path, cfg)
    anchor_rgb = np.asarray(rgb[anchor_frame])
    anchor_mask = _load_anchor_object_mask(ep_path, cfg)
    depth_anchor = np.asarray(depth_raw[anchor_frame]).astype(np.float32) * float(depth_scale)
    rng = np.random.default_rng(int(_cfg_get(cfg, "contact_heatmap.seed", 42)))
    pc = build_anchor_point_cloud(
        depth_anchor,
        anchor_mask,
        intrinsics,
        num_points=int(_cfg_get(cfg, "contact_heatmap.num_points", 4096)),
        rng=rng,
        outlier_std_ratio=float(_cfg_get(cfg, "contact_heatmap.outlier_std_ratio", 2.5)),
        normal_k=int(_cfg_get(cfg, "contact_heatmap.normal_k", 16)),
    )

    transforms = _load_se3_relative(ep_path)
    point_evidence_frames = []
    frame_stats = []
    aligned_visual_values = []
    for frame in frames:
        hand_mask = _load_hand_mask_for_frame(ep_path, cfg, frame)
        object_mask_t, object_source = _load_object_mask_for_frame(ep_path, cfg, frame, anchor_mask)
        sigma_px = _bbox_sigma(object_mask_t, cfg)
        evidence_img, dist_img = _frame_pixel_evidence(hand_mask, object_mask_t, sigma_px)

        pts_frame, transform_source = _transform_anchor_points(pc.points_camera, transforms, anchor_frame, frame)
        uv = project_to_2d(pts_frame, intrinsics)
        u, v = uv[:, 0], uv[:, 1]
        h, w = evidence_img.shape
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(pts_frame[:, 2]) & (pts_frame[:, 2] > 0)
        point_e = np.zeros(len(pc.points_camera), dtype=np.float32)
        point_e[valid] = evidence_img[v[valid], u[valid]].astype(np.float32)
        point_evidence_frames.append(point_e)
        aligned_visual_values.append(point_e)

        frame_stats.append(
            {
                "frame": int(frame),
                "sigma_px": float(sigma_px),
                "object_mask_source": object_source,
                "transform_source": transform_source,
                "point_valid_ratio": float(np.mean(valid)),
                "pixel_evidence_max": float(np.max(evidence_img)),
                "point_evidence_max": float(np.max(point_e)),
                "point_evidence_mean": float(np.mean(point_e)),
            }
        )
        _write_frame_evidence_visual(
            np.asarray(rgb[frame]),
            object_mask_t,
            hand_mask,
            evidence_img,
            viz_dir / f"frame_{frame:06d}_distance_evidence.png",
            f"frame {frame} distance evidence",
        )

    evidence_stack = np.stack(point_evidence_frames, axis=0).astype(np.float32)
    point_evidence_agg, active_count = _aggregate_topk(evidence_stack, cfg)
    seed_mask = (point_evidence_agg >= float(_cfg_get(cfg, "contact_heatmap.seed_threshold", 0.45))) & anchor_mask[
        pc.pixels_uv[:, 1], pc.pixels_uv[:, 0]
    ]
    if int(np.sum(seed_mask)) < 3 and float(np.max(point_evidence_agg)) > 0:
        topk = min(max(3, int(_cfg_get(cfg, "contact_heatmap.fallback_topk", 48))), len(point_evidence_agg))
        idx = np.argsort(point_evidence_agg)[-topk:]
        seed_mask = np.zeros(len(point_evidence_agg), dtype=bool)
        seed_mask[idx] = point_evidence_agg[idx] > 0
    seed_mask, seed_component_meta = _point_seed_components(pc.pixels_uv, seed_mask, anchor_mask.shape, cfg)

    heatmap, ellipse_meta = _fit_weighted_ellipse_heatmap(pc.pixels_uv, point_evidence_agg, anchor_mask, seed_mask, cfg)
    raw_point_heat = heatmap[pc.pixels_uv[:, 1], pc.pixels_uv[:, 0]].astype(np.float32)

    correction_cfg = _cfg_get(cfg, "contact_heatmap.surface_correction", {})
    if bool(correction_cfg.get("enabled", True)):
        point_heat, correction_meta = _connected_components_high_heat(
            pc.points_object_m,
            pc.normals_camera,
            raw_point_heat,
            threshold=float(correction_cfg.get("threshold", 0.4)),
            k=int(correction_cfg.get("knn", 12)),
            max_neighbor_dist_m=float(correction_cfg.get("max_neighbor_dist_m", 0.025)),
            min_normal_dot=float(correction_cfg.get("min_normal_dot", 0.2)),
        )
    else:
        point_heat = raw_point_heat
        correction_meta = {"enabled": False}

    corrected_heatmap_sparse = np.zeros_like(heatmap, dtype=np.float32)
    corrected_heatmap_sparse[pc.pixels_uv[:, 1], pc.pixels_uv[:, 0]] = point_heat

    meta = {
        "episode": ep_path.name,
        "anchor_frame": int(anchor_frame),
        "contact_start": timing.get("contact_start"),
        "contact_end": timing.get("contact_end"),
        "used_contact_frames": [int(f) for f in frames],
        "point_count": int(len(pc.points_camera)),
        "valid_depth_ratio": float(pc.valid_depth_ratio),
        "seed_count": int(np.sum(seed_mask)),
        "heat_area_ratio": float(np.mean(heatmap > float(_cfg_get(cfg, "contact_heatmap.heat_area_threshold", 0.2)))),
        "raw_point_heat_max": float(np.max(raw_point_heat)) if len(raw_point_heat) else 0.0,
        "corrected_point_heat_max": float(np.max(point_heat)) if len(point_heat) else 0.0,
        "frame_stats": frame_stats,
        "seed_components": seed_component_meta,
        "ellipse": ellipse_meta,
        "surface_correction": correction_meta,
    }

    np.savez_compressed(
        out_path,
        points_camera=pc.points_camera,
        points_object_m=pc.points_object_m,
        points_object_norm=pc.points_object_norm,
        normals_camera=pc.normals_camera,
        pixels_uv=pc.pixels_uv,
        object_center_camera=pc.object_center_camera.astype(np.float32),
        object_scale=np.asarray(pc.object_scale, dtype=np.float32),
        anchor_object_mask=anchor_mask.astype(np.uint8),
        per_frame_point_evidence=evidence_stack,
        aggregated_point_evidence=point_evidence_agg.astype(np.float32),
        active_frame_count=active_count.astype(np.int32),
        seed_mask=seed_mask.astype(np.uint8),
        heatmap_2d=heatmap.astype(np.float32),
        contact_heat_raw=raw_point_heat.astype(np.float32),
        contact_heat=point_heat.astype(np.float32),
        corrected_sparse_heatmap=corrected_heatmap_sparse.astype(np.float32),
        contact_frames=np.asarray(frames, dtype=np.int32),
    )
    with (out_dir / "contact_heatmap_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for frame, values in zip(frames, aligned_visual_values):
        _write_point_evidence_visual(
            anchor_rgb,
            anchor_mask,
            pc.pixels_uv,
            values,
            viz_dir / f"frame_{frame:06d}_aligned_point_evidence.png",
            f"frame {frame} aligned to anchor",
        )
    _write_point_evidence_visual(anchor_rgb, anchor_mask, pc.pixels_uv, point_evidence_agg, viz_dir / "aggregated_point_evidence.png", "Top-K aggregated evidence")
    _write_point_evidence_visual(anchor_rgb, anchor_mask, pc.pixels_uv[seed_mask], point_evidence_agg[seed_mask], viz_dir / "contact_seeds.png", "contact seeds")
    _write_heatmap_visual(anchor_rgb, heatmap, viz_dir / "final_2d_elliptical_heatmap.png", "final 2D elliptical heatmap")
    _write_point_evidence_visual(anchor_rgb, anchor_mask, pc.pixels_uv, raw_point_heat, viz_dir / "raw_point_heat_on_anchor.png", "raw point heat before 3D correction")
    _write_point_evidence_visual(anchor_rgb, anchor_mask, pc.pixels_uv, point_heat, viz_dir / "corrected_point_heat_on_anchor.png", "point heat after 3D correction")
    _write_3d_visual(pc.points_object_m, raw_point_heat, viz_dir / "raw_point_heat_3d.png")
    _write_3d_visual(pc.points_object_m, point_heat, viz_dir / "corrected_point_heat_3d.png")

    print(
        f"[contact_heatmap] {ep_path.name}: frames={frames} seeds={meta['seed_count']} "
        f"heat_area={meta['heat_area_ratio']:.4f} center={ellipse_meta.get('center_uv')}"
    )
    return True


def run(cfg) -> None:
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg)
        except Exception as exc:
            print(f"[contact_heatmap] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "contact_heatmap_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[contact_heatmap] failed {len(failed)} episodes; wrote {log_path}")
