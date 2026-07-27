from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial import cKDTree

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.pipeline.tracking import load_episode_camera_params, project_to_2d
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


@dataclass(frozen=True)
class AnchorPointCloud:
    points_camera: np.ndarray
    pixels_uv: np.ndarray
    normals_camera: np.ndarray
    object_center_camera: np.ndarray
    points_object_m: np.ndarray
    points_object_norm: np.ndarray
    object_scale: float
    valid_depth_ratio: float


def _cfg_get(cfg, dotted_key: str, default=None):
    cur = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    return np.asarray(mask > 0.5, dtype=bool)


def _load_anchor_object_mask(ep_path: Path, cfg, anchor_frame: int) -> np.ndarray:
    contact_cfg = _cfg_get(cfg, "contact", {})
    mask_rel = contact_cfg.get("object_mask_path", "sam_mask/affordance_mask.npy")
    mask_path = ep_path / str(mask_rel)
    masks_dir = contact_cfg.get("object_masks_dir")

    if masks_dir:
        pattern = str(contact_cfg.get("object_mask_pattern", "frame_{frame:06d}.npy"))
        frame_mask = ep_path / str(masks_dir) / pattern.format(frame=anchor_frame)
        if frame_mask.exists():
            return _as_bool_mask(np.load(frame_mask, allow_pickle=True))

    if anchor_frame != 0 and not bool(contact_cfg.get("allow_first_frame_mask_for_anchor", False)):
        raise ValueError(
            f"anchor_frame={anchor_frame} but only first-frame object mask exists. "
            "Set contact.object_masks_dir or contact.allow_first_frame_mask_for_anchor=true."
        )
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing anchor object mask: {mask_path}")
    return _as_bool_mask(np.load(mask_path, allow_pickle=True))


def _unproject_pixels(uv: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = uv[:, 0].astype(np.int64)
    v = uv[:, 1].astype(np.int64)
    valid = (
        (u >= 0)
        & (u < depth_m.shape[1])
        & (v >= 0)
        & (v < depth_m.shape[0])
    )
    z = np.zeros(len(uv), dtype=np.float32)
    z[valid] = depth_m[v[valid], u[valid]].astype(np.float32)
    valid &= np.isfinite(z) & (z > 0)
    u_valid = u[valid].astype(np.float32)
    v_valid = v[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    pts = np.stack([x, y, z_valid], axis=-1).astype(np.float32)
    return pts, valid


def _statistical_depth_filter(points: np.ndarray, std_ratio: float) -> np.ndarray:
    if len(points) < 16 or std_ratio <= 0:
        return np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    k = min(12, len(points))
    dists, _ = tree.query(points, k=k)
    mean_d = np.mean(dists[:, 1:], axis=1)
    thresh = float(np.mean(mean_d) + std_ratio * np.std(mean_d))
    return mean_d <= thresh


def _estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    if len(points) < 4:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))
    tree = cKDTree(points)
    k = min(max(4, k), len(points))
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points, dtype=np.float32)
    view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for i, neigh_idx in enumerate(idx):
        neigh = points[neigh_idx]
        centered = neigh - np.mean(neigh, axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(neigh) - 1, 1)
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, 0].astype(np.float32)
        if np.dot(n, view_dir) < 0:
            n = -n
        normals[i] = n / max(float(np.linalg.norm(n)), 1e-8)
    return normals


def build_anchor_point_cloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    num_points: int,
    rng: np.random.Generator,
    outlier_std_ratio: float,
    normal_k: int,
) -> AnchorPointCloud:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Anchor object mask is empty.")
    uv_all = np.stack([xs, ys], axis=-1).astype(np.int32)
    pts_all, valid_depth = _unproject_pixels(uv_all, depth_m, intrinsics)
    valid_uv = uv_all[valid_depth]
    valid_depth_ratio = float(np.mean(valid_depth))
    if len(pts_all) == 0:
        raise ValueError("No valid depth inside anchor object mask.")

    keep = _statistical_depth_filter(pts_all, outlier_std_ratio)
    pts_all = pts_all[keep]
    valid_uv = valid_uv[keep]
    if len(pts_all) == 0:
        raise ValueError("All anchor object points were removed as outliers.")

    if len(pts_all) > num_points:
        idx = rng.choice(len(pts_all), size=num_points, replace=False)
        pts = pts_all[idx]
        uv = valid_uv[idx]
    elif len(pts_all) < num_points:
        pad = rng.choice(len(pts_all), size=num_points - len(pts_all), replace=True)
        pts = np.concatenate([pts_all, pts_all[pad]], axis=0)
        uv = np.concatenate([valid_uv, valid_uv[pad]], axis=0)
    else:
        pts = pts_all
        uv = valid_uv

    center = np.median(pts, axis=0).astype(np.float32)
    points_object_m = (pts - center[None, :]).astype(np.float32)
    scale = float(np.percentile(np.linalg.norm(points_object_m, axis=1), 95))
    scale = max(scale, 1e-6)
    points_object_norm = (points_object_m / scale).astype(np.float32)
    normals = _estimate_normals(pts.astype(np.float32), normal_k)
    return AnchorPointCloud(
        points_camera=pts.astype(np.float32),
        pixels_uv=uv.astype(np.int32),
        normals_camera=normals.astype(np.float32),
        object_center_camera=center,
        points_object_m=points_object_m,
        points_object_norm=points_object_norm,
        object_scale=scale,
        valid_depth_ratio=valid_depth_ratio,
    )


def _contact_frames(cfg) -> list[int]:
    contact_cfg = _cfg_get(cfg, "contact", {})
    if "contact_frames" in contact_cfg:
        return [int(x) for x in contact_cfg["contact_frames"]]
    window = contact_cfg.get("contact_window", [0, 0])
    if len(window) != 2:
        raise ValueError("contact.contact_window must be [start, end], inclusive.")
    start, end = int(window[0]), int(window[1])
    if end < start:
        raise ValueError(f"Invalid contact window: {window}")
    stride = int(contact_cfg.get("contact_frame_stride", 1))
    return list(range(start, end + 1, max(stride, 1)))


def _load_bbox_for_frame(ep_path: Path, cfg, frame: int) -> tuple[np.ndarray | None, str]:
    contact_cfg = _cfg_get(cfg, "contact", {})
    by_episode = contact_cfg.get("hand_bboxes_by_episode", {})
    by_frame = contact_cfg.get("hand_bboxes_by_frame", {})
    ep_boxes = by_episode.get(ep_path.name, {}) if isinstance(by_episode, dict) else {}

    for source in (ep_boxes, by_frame):
        if not isinstance(source, dict):
            continue
        val = source.get(str(frame), source.get(frame))
        if val is not None:
            arr = np.asarray(val, dtype=np.float32).reshape(4)
            return arr, "bbox"
    default_bbox = contact_cfg.get("hand_bbox")
    if default_bbox is not None:
        arr = np.asarray(default_bbox, dtype=np.float32).reshape(4)
        return arr, "bbox_default"
    return None, "missing"


def _load_hand_mask_for_frame(ep_path: Path, cfg, frame: int, image_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    contact_cfg = _cfg_get(cfg, "contact", {})
    masks_dir = contact_cfg.get("hand_masks_dir")
    if masks_dir:
        pattern = str(contact_cfg.get("hand_mask_pattern", "frame_{frame:06d}.npy"))
        path = ep_path / str(masks_dir) / pattern.format(frame=frame)
        if path.exists():
            return _as_bool_mask(np.load(path, allow_pickle=True)), "mask"

    bbox, source = _load_bbox_for_frame(ep_path, cfg, frame)
    if bbox is None:
        raise FileNotFoundError(
            f"Missing hand mask/bbox for {ep_path.name} frame {frame}. "
            "Set contact.hand_masks_dir or contact.hand_bboxes_by_frame/hand_bbox."
        )
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    x0 = int(np.clip(np.floor(x0), 0, w - 1))
    x1 = int(np.clip(np.ceil(x1), 0, w))
    y0 = int(np.clip(np.floor(y0), 0, h - 1))
    y1 = int(np.clip(np.ceil(y1), 0, h))
    mask = np.zeros((h, w), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid hand bbox for frame {frame}: {bbox.tolist()}")
    mask[y0:y1, x0:x1] = True
    return mask, source


def _transform_anchor_points(points_anchor: np.ndarray, transforms_0_to_t: np.ndarray | None, anchor_frame: int, frame: int):
    if transforms_0_to_t is None:
        return points_anchor.copy(), "identity"
    if frame >= len(transforms_0_to_t) or anchor_frame >= len(transforms_0_to_t):
        raise IndexError(
            f"SE(3) trajectory has {len(transforms_0_to_t)} frames, cannot use anchor={anchor_frame}, frame={frame}."
        )
    T_0_to_anchor = transforms_0_to_t[anchor_frame]
    T_0_to_frame = transforms_0_to_t[frame]
    T_anchor_to_frame = T_0_to_frame @ np.linalg.inv(T_0_to_anchor)
    pts_h = np.concatenate([points_anchor, np.ones((len(points_anchor), 1), dtype=np.float32)], axis=1)
    return (T_anchor_to_frame @ pts_h.T).T[:, :3].astype(np.float32), "se3_relative"


def _load_se3_relative(ep_path: Path) -> np.ndarray | None:
    path = ep_path / "se3_trajectory" / "se3_relative_trajectory.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "T_cam_0_to_t" not in data:
        return None
    return data["T_cam_0_to_t"].astype(np.float32)


def aggregate_contact_evidence(
    points_anchor: np.ndarray,
    depth_video_m: np.ndarray,
    intrinsics: np.ndarray,
    transforms_0_to_t: np.ndarray | None,
    anchor_frame: int,
    contact_frames: Iterable[int],
    cfg,
    ep_path: Path,
) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    contact_cfg = _cfg_get(cfg, "contact", {})
    sigma_px = float(contact_cfg.get("distance_sigma_px", 8.0))
    depth_sigma_m = float(contact_cfg.get("depth_sigma_m", 0.025))
    use_depth = bool(contact_cfg.get("use_depth_consistency", True))
    evidence = np.zeros(len(points_anchor), dtype=np.float32)
    frame_stats = []
    reproj_frames = []
    h, w = depth_video_m.shape[1:]

    for frame in contact_frames:
        if frame < 0 or frame >= depth_video_m.shape[0]:
            raise IndexError(f"contact frame {frame} is outside video length {depth_video_m.shape[0]}")
        hand_mask, hand_source = _load_hand_mask_for_frame(ep_path, cfg, frame, (h, w))
        inv_hand = (~hand_mask).astype(np.uint8)
        dist_map = cv2.distanceTransform(inv_hand, cv2.DIST_L2, 5).astype(np.float32)
        pts_frame, transform_source = _transform_anchor_points(points_anchor, transforms_0_to_t, anchor_frame, frame)
        uv = project_to_2d(pts_frame, intrinsics)
        u = uv[:, 0]
        v = uv[:, 1]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(pts_frame[:, 2]) & (pts_frame[:, 2] > 0)
        frame_evidence = np.zeros(len(points_anchor), dtype=np.float32)
        if np.any(valid):
            d_px = dist_map[v[valid], u[valid]]
            frame_evidence[valid] = np.exp(-(d_px ** 2) / (2.0 * sigma_px ** 2)).astype(np.float32)
            if use_depth:
                obs_depth = depth_video_m[frame, v[valid], u[valid]].astype(np.float32)
                depth_valid = np.isfinite(obs_depth) & (obs_depth > 0)
                depth_w = np.ones_like(obs_depth, dtype=np.float32)
                depth_w[depth_valid] = np.exp(
                    -(np.abs(obs_depth[depth_valid] - pts_frame[valid][depth_valid, 2]) ** 2)
                    / (2.0 * depth_sigma_m ** 2)
                ).astype(np.float32)
                frame_evidence[valid] *= depth_w
        evidence = np.maximum(evidence, frame_evidence)
        frame_stats.append(
            {
                "frame": int(frame),
                "hand_source": hand_source,
                "transform_source": transform_source,
                "projected_valid_ratio": float(np.mean(valid)),
                "mean_evidence": float(np.mean(frame_evidence)),
                "max_evidence": float(np.max(frame_evidence)) if len(frame_evidence) else 0.0,
            }
        )
        reproj_frames.append(np.stack([uv[:, 0], uv[:, 1], frame_evidence], axis=-1).astype(np.float32))

    return evidence, {"frames": frame_stats}, reproj_frames


def fit_elliptical_heatmap(
    pixels_uv: np.ndarray,
    evidence: np.ndarray,
    object_mask: np.ndarray,
    *,
    seed_threshold: float,
    fallback_topk: int,
    min_axis_px: float,
    max_axis_px: float,
    cov_regularizer: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = object_mask.shape
    weights = evidence.astype(np.float64)
    seed_mask = weights >= seed_threshold
    if int(np.sum(seed_mask)) < 3 and np.max(weights) > 0:
        topk = min(max(3, fallback_topk), len(weights))
        idx = np.argsort(weights)[-topk:]
        seed_mask = np.zeros(len(weights), dtype=bool)
        seed_mask[idx] = True
    if int(np.sum(seed_mask)) < 3:
        return np.zeros((h, w), dtype=np.float32), seed_mask, {
            "status": "empty",
            "seed_count": int(np.sum(seed_mask)),
            "reason": "fewer than 3 contact seeds",
        }

    seed_uv = pixels_uv[seed_mask].astype(np.float64)
    seed_w = np.maximum(weights[seed_mask], 1e-6)
    seed_w = seed_w / np.sum(seed_w)
    center = np.sum(seed_uv * seed_w[:, None], axis=0)
    delta = seed_uv - center[None, :]
    cov = (delta * seed_w[:, None]).T @ delta
    cov += np.eye(2, dtype=np.float64) * cov_regularizer
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, min_axis_px ** 2, max_axis_px ** 2)
    cov = vecs @ np.diag(vals) @ vecs.T
    inv_cov = np.linalg.inv(cov)

    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.stack([xx - center[0], yy - center[1]], axis=-1).astype(np.float64)
    maha = np.einsum("...i,ij,...j->...", grid, inv_cov, grid)
    heat = np.exp(-0.5 * maha).astype(np.float32)
    heat *= object_mask.astype(np.float32)
    if float(np.max(heat)) > 0:
        heat /= float(np.max(heat))
    meta = {
        "status": "ok",
        "seed_count": int(np.sum(seed_mask)),
        "center_uv": center.astype(float).tolist(),
        "axis_std_px": np.sqrt(vals).astype(float).tolist(),
        "covariance": cov.astype(float).tolist(),
    }
    return heat.astype(np.float32), seed_mask, meta


def _connected_components_high_heat(
    points: np.ndarray,
    normals: np.ndarray,
    heat: np.ndarray,
    *,
    threshold: float,
    k: int,
    max_neighbor_dist_m: float,
    min_normal_dot: float,
) -> tuple[np.ndarray, dict]:
    high_idx = np.flatnonzero(heat >= threshold)
    if len(high_idx) < 3:
        return heat.astype(np.float32), {"enabled": True, "status": "skipped", "reason": "too few high-heat points"}
    high_points = points[high_idx]
    tree = cKDTree(high_points)
    k = min(max(2, k), len(high_idx))
    dists, neigh = tree.query(high_points, k=k)
    visited = np.zeros(len(high_idx), dtype=bool)
    components = []
    for start in range(len(high_idx)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for dist, nb in zip(np.atleast_1d(dists[cur])[1:], np.atleast_1d(neigh[cur])[1:]):
                if visited[nb] or dist > max_neighbor_dist_m:
                    continue
                n0 = normals[high_idx[cur]]
                n1 = normals[high_idx[nb]]
                if float(np.dot(n0, n1)) < min_normal_dot:
                    continue
                visited[nb] = True
                queue.append(int(nb))
        components.append(np.asarray(comp, dtype=np.int64))
    best_comp = max(components, key=lambda c: float(np.max(heat[high_idx[c]])))
    keep_high = high_idx[best_comp]
    corrected = heat.copy()
    drop_high = np.setdiff1d(high_idx, keep_high, assume_unique=False)
    corrected[drop_high] = 0.0
    if float(np.max(corrected)) > 0:
        corrected = corrected / float(np.max(corrected))
    return corrected.astype(np.float32), {
        "enabled": True,
        "status": "ok",
        "num_components": int(len(components)),
        "kept_points": int(len(keep_high)),
        "dropped_points": int(len(drop_high)),
        "high_points": int(len(high_idx)),
    }


def _quality_label(meta: dict, cfg) -> str:
    qcfg = _cfg_get(cfg, "contact.quality", {})
    if meta["point_count"] <= 0 or meta["seed_count"] <= 0 or meta["heat_max"] <= 0:
        return "reject"
    if meta["valid_depth_ratio"] < float(qcfg.get("min_valid_depth_ratio_good", 0.35)):
        return "review"
    if meta["seed_count"] < int(qcfg.get("min_seed_count_good", 8)):
        return "review"
    area = meta["heat_area_ratio"]
    if area <= 0 or area > float(qcfg.get("max_heat_area_ratio_good", 0.35)):
        return "review"
    return "good"


def _write_anchor_visual(rgb: np.ndarray, object_mask: np.ndarray, out_path: Path) -> None:
    overlay = rgb.copy()
    overlay[object_mask] = (0.55 * overlay[object_mask] + np.array([255, 220, 0]) * 0.45).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _write_seed_visual(rgb: np.ndarray, object_mask: np.ndarray, pixels_uv: np.ndarray, evidence: np.ndarray, seed_mask: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.contour(object_mask, colors="yellow", linewidths=1.0)
    if np.any(evidence > 0):
        sc = ax.scatter(pixels_uv[:, 0], pixels_uv[:, 1], c=evidence, s=8, cmap="magma", vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, fraction=0.035)
    if np.any(seed_mask):
        ax.scatter(pixels_uv[seed_mask, 0], pixels_uv[seed_mask, 1], c="cyan", s=12, marker="x")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_heatmap_visual(rgb: np.ndarray, heatmap: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.imshow(heatmap, cmap="magma", alpha=np.clip(heatmap, 0, 1) * 0.75, vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_3d_visual(points_object: np.ndarray, heat: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points_object[:, 0], points_object[:, 1], points_object[:, 2], c=heat, s=5, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, fraction=0.035)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=25, azim=-60)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_reprojection_visual(rgb_video: np.ndarray, reproj_frames: list[np.ndarray], contact_frames: list[int], out_path: Path) -> None:
    cols = min(4, len(contact_frames))
    rows = int(np.ceil(len(contact_frames) / max(cols, 1)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, frame, reproj in zip(axes, contact_frames, reproj_frames):
        ax.imshow(rgb_video[frame])
        if len(reproj):
            keep = reproj[:, 2] > 0.05
            ax.scatter(reproj[keep, 0], reproj[keep, 1], c=reproj[keep, 2], s=5, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"frame {frame}")
        ax.axis("off")
    for ax in axes[len(contact_frames):]:
        ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def process_episode(ep_path: str | Path, cfg) -> bool:
    ep_path = Path(ep_path)
    contact_cfg = _cfg_get(cfg, "contact", {})
    out_dir = ep_path / str(contact_cfg.get("output_dir", "contact_field"))
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "contact_field.npz"
    if npz_path.exists() and not bool(_cfg_get(cfg, "runtime.overwrite", False)):
        print(f"[contact] skip existing {ep_path.name}")
        return True

    anchor_frame = int(contact_cfg.get("anchor_frame", 0))
    frames = _contact_frames(cfg)
    rng = np.random.default_rng(int(contact_cfg.get("seed", 42)))

    rgb = zarr.open(str(ep_path / "rgb"), mode="r")[:]
    depth_raw = zarr.open(str(ep_path / "depth"), mode="r")[:]
    intrinsics, depth_scale, _meta = load_episode_camera_params(ep_path, cfg)
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    if anchor_frame < 0 or anchor_frame >= rgb.shape[0]:
        raise IndexError(f"anchor_frame {anchor_frame} outside video length {rgb.shape[0]}")

    object_mask = _load_anchor_object_mask(ep_path, cfg, anchor_frame)
    pc = build_anchor_point_cloud(
        depth_m[anchor_frame],
        object_mask,
        intrinsics,
        num_points=int(contact_cfg.get("num_points", 4096)),
        rng=rng,
        outlier_std_ratio=float(contact_cfg.get("outlier_std_ratio", 2.5)),
        normal_k=int(contact_cfg.get("normal_k", 16)),
    )

    transforms = _load_se3_relative(ep_path)
    evidence, evidence_meta, reproj_frames = aggregate_contact_evidence(
        pc.points_camera,
        depth_m,
        intrinsics,
        transforms,
        anchor_frame,
        frames,
        cfg,
        ep_path,
    )
    heatmap, seed_mask, gaussian_meta = fit_elliptical_heatmap(
        pc.pixels_uv,
        evidence,
        object_mask,
        seed_threshold=float(contact_cfg.get("seed_threshold", 0.5)),
        fallback_topk=int(contact_cfg.get("fallback_topk", 32)),
        min_axis_px=float(contact_cfg.get("min_axis_px", 6.0)),
        max_axis_px=float(contact_cfg.get("max_axis_px", 80.0)),
        cov_regularizer=float(contact_cfg.get("cov_regularizer", 9.0)),
    )

    u = pc.pixels_uv[:, 0]
    v = pc.pixels_uv[:, 1]
    contact_heat_raw = heatmap[v, u].astype(np.float32)
    correction_cfg = contact_cfg.get("surface_correction", {})
    if bool(correction_cfg.get("enabled", True)):
        contact_heat, correction_meta = _connected_components_high_heat(
            pc.points_object_m,
            pc.normals_camera,
            contact_heat_raw,
            threshold=float(correction_cfg.get("threshold", 0.4)),
            k=int(correction_cfg.get("knn", 12)),
            max_neighbor_dist_m=float(correction_cfg.get("max_neighbor_dist_m", 0.025)),
            min_normal_dot=float(correction_cfg.get("min_normal_dot", 0.2)),
        )
    else:
        contact_heat = contact_heat_raw
        correction_meta = {"enabled": False}

    meta = {
        "episode": ep_path.name,
        "anchor_frame": int(anchor_frame),
        "contact_frames": [int(f) for f in frames],
        "point_count": int(len(pc.points_camera)),
        "object_mask_pixels": int(np.sum(object_mask)),
        "valid_depth_ratio": pc.valid_depth_ratio,
        "object_center_camera": pc.object_center_camera.astype(float).tolist(),
        "object_scale": float(pc.object_scale),
        "seed_count": int(np.sum(seed_mask)),
        "heat_max": float(np.max(contact_heat)) if len(contact_heat) else 0.0,
        "heat_mean": float(np.mean(contact_heat)) if len(contact_heat) else 0.0,
        "heat_area_ratio": float(np.mean(heatmap > float(contact_cfg.get("heat_area_threshold", 0.2)))),
        "evidence": evidence_meta,
        "gaussian": gaussian_meta,
        "surface_correction": correction_meta,
        "dino_features": {"enabled": False, "reason": "DINOv2 dense feature extraction is not implemented in MVP."},
    }
    meta["quality"] = _quality_label(meta, cfg)

    np.savez_compressed(
        npz_path,
        points_camera=pc.points_camera,
        points_object_m=pc.points_object_m,
        points_object_norm=pc.points_object_norm,
        pixels_uv=pc.pixels_uv,
        normals_camera=pc.normals_camera,
        contact_evidence=evidence.astype(np.float32),
        contact_heat_raw=contact_heat_raw.astype(np.float32),
        contact_heat=contact_heat.astype(np.float32),
        heatmap_2d=heatmap.astype(np.float32),
        object_mask_anchor=object_mask.astype(np.uint8),
        seed_mask=seed_mask.astype(np.uint8),
        object_center_camera=pc.object_center_camera.astype(np.float32),
        object_scale=np.asarray(pc.object_scale, dtype=np.float32),
        anchor_frame=np.asarray(anchor_frame, dtype=np.int32),
        contact_frames=np.asarray(frames, dtype=np.int32),
    )
    with (out_dir / "contact_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _write_anchor_visual(rgb[anchor_frame], object_mask, viz_dir / "viz_anchor_mask.png")
    _write_seed_visual(rgb[anchor_frame], object_mask, pc.pixels_uv, evidence, seed_mask, viz_dir / "viz_contact_seeds.png")
    _write_heatmap_visual(rgb[anchor_frame], heatmap, viz_dir / "viz_contact_heatmap.png")
    _write_3d_visual(pc.points_object_m, contact_heat, viz_dir / "viz_contact_points_3d.png")
    _write_reprojection_visual(rgb, reproj_frames, frames, viz_dir / "viz_reprojection.png")
    print(f"[contact] {ep_path.name}: points={len(pc.points_camera)} seeds={meta['seed_count']} quality={meta['quality']}")
    return True


def run(cfg) -> None:
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, _cfg_get(cfg, "runtime.episodes", None)):
        try:
            process_episode(ep_path, cfg)
        except Exception as exc:
            print(f"[contact] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "contact_failed_logs.txt"
        try:
            with log_path.open("w", encoding="utf-8") as f:
                for ep, err in failed:
                    f.write(f"{ep}: {err}\n")
            print(f"[contact] failed {len(failed)} episodes; wrote {log_path}")
        except OSError as exc:
            print(f"[contact] failed {len(failed)} episodes; could not write {log_path}: {exc}")
