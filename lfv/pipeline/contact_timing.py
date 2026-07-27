from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.pipeline.tracking import load_episode_camera_params
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def _cfg_get(cfg, dotted_key: str, default=None):
    cur = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    return np.asarray(mask > 0.5, dtype=bool)


def _load_object_mask(ep_path: Path, cfg) -> np.ndarray:
    rel = str(_cfg_get(cfg, "contact_timing.object_mask_path", "sam_mask/affordance_mask.npy"))
    path = ep_path / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing object mask for contact timing: {path}")
    return _as_bool_mask(np.load(path, allow_pickle=True))


def _load_manual_timing(ep_path: Path, cfg) -> dict | None:
    manual = _cfg_get(cfg, "contact_timing.manual")
    if isinstance(manual, dict):
        ep_manual = manual.get(ep_path.name, manual.get("default"))
        if isinstance(ep_manual, dict):
            return {k: int(v) if k in {"anchor_frame", "contact_start", "contact_end"} else v for k, v in ep_manual.items()}
    return None


def _hand_mask_paths(ep_path: Path, cfg) -> list[tuple[int, Path]]:
    mask_dir = ep_path / str(_cfg_get(cfg, "contact_timing.hand_mask_dir", "hand_mask"))
    paths = []
    for path in sorted(mask_dir.glob("frame_*.npy")):
        stem = path.stem
        try:
            frame = int(stem.split("_")[-1])
        except ValueError:
            continue
        paths.append((frame, path))
    return paths


def _frame_metrics(hand_mask: np.ndarray, object_mask: np.ndarray, depth_m: np.ndarray | None = None) -> dict:
    hand = _as_bool_mask(hand_mask)
    obj = object_mask.astype(bool)
    obj_pixels = max(int(np.sum(obj)), 1)
    hand_pixels = int(np.sum(hand))
    overlap = hand & obj
    overlap_pixels = int(np.sum(overlap))
    overlap_ratio = float(overlap_pixels / obj_pixels)

    if hand_pixels == 0:
        min_dist = float("inf")
        p05_dist = float("inf")
        mean_obj_dist = float("inf")
    else:
        dist_to_hand = cv2.distanceTransform((~hand).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        obj_dist = dist_to_hand[obj]
        min_dist = float(np.min(obj_dist)) if len(obj_dist) else float("inf")
        p05_dist = float(np.percentile(obj_dist, 5)) if len(obj_dist) else float("inf")
        mean_obj_dist = float(np.mean(obj_dist)) if len(obj_dist) else float("inf")

    depth_valid_ratio = None
    if depth_m is not None:
        obj_depth = depth_m[obj]
        depth_valid_ratio = float(np.mean(np.isfinite(obj_depth) & (obj_depth > 0))) if len(obj_depth) else 0.0

    return {
        "hand_pixels": hand_pixels,
        "object_pixels": obj_pixels,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio": overlap_ratio,
        "min_distance_px": min_dist,
        "p05_distance_px": p05_dist,
        "mean_object_distance_px": mean_obj_dist,
        "object_depth_valid_ratio": depth_valid_ratio,
    }


def _is_contact(metrics: dict, cfg) -> bool:
    dist_thr = float(_cfg_get(cfg, "contact_timing.contact_distance_px", 8.0))
    overlap_thr = float(_cfg_get(cfg, "contact_timing.contact_overlap_ratio", 0.003))
    return (
        metrics["min_distance_px"] <= dist_thr
        or metrics["p05_distance_px"] <= dist_thr
        or metrics["overlap_ratio"] >= overlap_thr
    )


def _select_first_stable_contact(frame_metrics: list[dict], cfg) -> int | None:
    min_consecutive = max(1, int(_cfg_get(cfg, "contact_timing.min_consecutive", 2)))
    run = []
    for item in frame_metrics:
        if item["is_contact"]:
            run.append(item)
            if len(run) >= min_consecutive:
                return int(run[0]["frame"])
        else:
            run = []
    return None


def _select_anchor(frame_metrics: list[dict], contact_start: int, cfg) -> int:
    min_gap = max(0, int(_cfg_get(cfg, "contact_timing.anchor_min_gap_frames", 3)))
    min_dist = float(_cfg_get(cfg, "contact_timing.anchor_min_distance_px", 18.0))
    max_overlap = float(_cfg_get(cfg, "contact_timing.anchor_max_overlap_ratio", 0.001))
    min_depth = float(_cfg_get(cfg, "contact_timing.min_object_depth_valid_ratio", 0.2))
    candidates = []
    for item in frame_metrics:
        frame = int(item["frame"])
        if frame > contact_start - min_gap:
            continue
        depth_ratio = item.get("object_depth_valid_ratio")
        depth_ok = True if depth_ratio is None else float(depth_ratio) >= min_depth
        if item["min_distance_px"] >= min_dist and item["overlap_ratio"] <= max_overlap and depth_ok:
            candidates.append(item)
    if candidates:
        return int(candidates[-1]["frame"])
    return max(0, int(contact_start) - min_gap)


def _write_timing_visual(ep_path: Path, rgb, object_mask: np.ndarray, frame_metrics: list[dict], timing: dict, out_path: Path) -> None:
    frames = [timing["anchor_frame"], timing["contact_start"], timing["contact_end"]]
    titles = ["anchor", "contact_start", "contact_end"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    hand_dir = ep_path / "hand_mask"
    for ax, frame, title in zip(axes, frames, titles):
        frame = int(np.clip(frame, 0, rgb.shape[0] - 1))
        ax.imshow(rgb[frame])
        ax.contour(object_mask, colors="yellow", linewidths=1.0)
        mask_path = hand_dir / f"frame_{frame:06d}.npy"
        if not mask_path.exists():
            detected_frames = np.asarray([m["frame"] for m in frame_metrics], dtype=np.int64)
            if len(detected_frames):
                nearest = int(detected_frames[np.argmin(np.abs(detected_frames - frame))])
                mask_path = hand_dir / f"frame_{nearest:06d}.npy"
        if mask_path.exists():
            hand = _as_bool_mask(np.load(mask_path, allow_pickle=True))
            ax.imshow(hand, cmap="autumn", alpha=hand.astype(np.float32) * 0.45)
        ax.set_title(f"{title}: {frame}")
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def process_episode(ep_path: str | Path, cfg) -> bool:
    ep_path = Path(ep_path)
    out_dir = ep_path / str(_cfg_get(cfg, "contact_timing.output_dir", "contact_timing"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "contact_timing.json"
    overwrite = bool(_cfg_get(cfg, "runtime.overwrite", False))
    if out_path.exists() and not overwrite:
        print(f"[timing] skip existing {ep_path.name}")
        return True

    manual = _load_manual_timing(ep_path, cfg)
    rgb = zarr.open(str(ep_path / "rgb"), mode="r")
    depth_raw = zarr.open(str(ep_path / "depth"), mode="r")
    _intrinsics, depth_scale, _meta = load_episode_camera_params(ep_path, cfg)
    object_mask = _load_object_mask(ep_path, cfg)

    paths = _hand_mask_paths(ep_path, cfg)
    frame_metrics = []
    for frame, mask_path in paths:
        depth_m = np.asarray(depth_raw[frame]).astype(np.float32) * float(depth_scale)
        metrics = _frame_metrics(np.load(mask_path, allow_pickle=True), object_mask, depth_m)
        item = {"frame": int(frame), **metrics}
        item["is_contact"] = bool(_is_contact(item, cfg))
        frame_metrics.append(item)

    if manual is not None:
        timing = {
            "episode": ep_path.name,
            "mode": "manual",
            "anchor_frame": int(manual["anchor_frame"]),
            "contact_start": int(manual["contact_start"]),
            "contact_end": int(manual["contact_end"]),
            "quality": str(manual.get("quality", "review")),
            "metrics": {"frames": frame_metrics},
        }
    else:
        if not frame_metrics:
            timing = {
                "episode": ep_path.name,
                "mode": "auto",
                "anchor_frame": 0,
                "contact_start": None,
                "contact_end": None,
                "quality": "reject",
                "reason": "no hand masks available",
                "metrics": {"frames": []},
            }
        else:
            contact_start = _select_first_stable_contact(frame_metrics, cfg)
            if contact_start is None:
                timing = {
                    "episode": ep_path.name,
                    "mode": "auto",
                    "anchor_frame": 0,
                    "contact_start": None,
                    "contact_end": None,
                    "quality": "reject",
                    "reason": "no stable hand-object contact found",
                    "metrics": {"frames": frame_metrics},
                }
            else:
                window_len = max(1, int(_cfg_get(cfg, "contact_timing.contact_window_frames", 8)))
                available_contact_frames = [m["frame"] for m in frame_metrics if int(m["frame"]) >= int(contact_start)]
                contact_end = int(available_contact_frames[min(window_len - 1, len(available_contact_frames) - 1)])
                anchor_frame = _select_anchor(frame_metrics, int(contact_start), cfg)
                timing = {
                    "episode": ep_path.name,
                    "mode": "auto",
                    "anchor_frame": int(anchor_frame),
                    "contact_start": int(contact_start),
                    "contact_end": int(contact_end),
                    "contact_frames": [int(f) for f in available_contact_frames[:window_len]],
                    "quality": "good",
                    "metrics": {"frames": frame_metrics},
                }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    if timing.get("contact_start") is not None:
        _write_timing_visual(
            ep_path,
            rgb,
            object_mask,
            frame_metrics,
            timing,
            out_dir / "contact_timing_overlay.png",
        )
    print(
        f"[timing] {ep_path.name}: quality={timing['quality']} "
        f"anchor={timing.get('anchor_frame')} contact={timing.get('contact_start')}..{timing.get('contact_end')}"
    )
    return True


def run(cfg) -> None:
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg)
        except Exception as exc:
            print(f"[timing] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "contact_timing_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[timing] failed {len(failed)} episodes; wrote {log_path}")
