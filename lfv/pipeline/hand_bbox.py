from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.pipeline.hand_segmentation import (
    _cfg_get,
    _detect_hand_bbox,
    _device,
    _frame_indices,
    _load_dino,
    _write_overlay,
)
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def _load_reject_masks(ep_path: Path, cfg) -> list[np.ndarray]:
    masks = []
    for rel in _cfg_get(cfg, "hand.reject_object_mask_paths", []):
        path = ep_path / str(rel)
        if path.exists():
            mask = np.load(path, allow_pickle=True)
            if mask.ndim == 2:
                masks.append(mask > 0.5)
    return masks


def _bbox_mask_overlap_ratio(bbox: np.ndarray, masks: list[np.ndarray]) -> float:
    if not masks:
        return 0.0
    h, w = masks[0].shape
    x0, y0, x1, y1 = bbox
    x0 = int(np.clip(np.floor(x0), 0, w - 1))
    x1 = int(np.clip(np.ceil(x1), 0, w))
    y0 = int(np.clip(np.floor(y0), 0, h - 1))
    y1 = int(np.clip(np.ceil(y1), 0, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = float((x1 - x0) * (y1 - y0))
    max_ratio = 0.0
    for mask in masks:
        max_ratio = max(max_ratio, float(np.sum(mask[y0:y1, x0:x1])) / max(area, 1.0))
    return max_ratio


def process_episode(ep_path: str | Path, cfg, processor, dino_model, device: str) -> bool:
    ep_path = Path(ep_path)
    bbox_dir = ep_path / str(_cfg_get(cfg, "hand.bbox_dir", "hand_bbox"))
    meta_dir = ep_path / str(_cfg_get(cfg, "hand.meta_dir", "hand_contact"))
    viz_dir = ep_path / str(_cfg_get(cfg, "hand.viz_dir", "viz"))
    bbox_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    rgb = zarr.open(str(ep_path / "rgb"), mode="r")
    frame_ids = _frame_indices(int(rgb.shape[0]), cfg)
    prompts = [str(_cfg_get(cfg, "hand.prompt", "hand ."))]
    prompts.extend(str(p) for p in _cfg_get(cfg, "hand.fallback_prompts", []))
    overwrite = bool(_cfg_get(cfg, "runtime.overwrite", False))
    viz_stride = max(1, int(_cfg_get(cfg, "hand.viz_stride", 12)))
    reject_masks = _load_reject_masks(ep_path, cfg)
    reject_overlap_thr = float(_cfg_get(cfg, "hand.reject_object_box_overlap_ratio", 0.45))

    frame_meta = []
    detected = 0
    for idx, frame in enumerate(frame_ids):
        bbox_path = bbox_dir / f"frame_{frame:06d}.npy"
        if bbox_path.exists() and not overwrite:
            detected += 1
            frame_meta.append({"frame": int(frame), "status": "skipped_existing"})
            continue

        frame_rgb = np.asarray(rgb[frame])
        bbox, det_meta = _detect_hand_bbox(frame_rgb, prompts, cfg, processor, dino_model, device)
        entry = {"frame": int(frame), **det_meta}
        if bbox is None:
            if overwrite and bbox_path.exists():
                bbox_path.unlink()
            frame_meta.append(entry)
            if idx % viz_stride == 0:
                _write_overlay(frame_rgb, None, None, f"{ep_path.name} frame {frame}: no hand bbox", viz_dir / f"hand_bbox_frame_{frame:06d}.png")
            continue
        object_overlap = _bbox_mask_overlap_ratio(bbox, reject_masks)
        entry["object_box_overlap_ratio"] = float(object_overlap)
        if object_overlap >= reject_overlap_thr:
            if overwrite and bbox_path.exists():
                bbox_path.unlink()
            entry["status"] = "rejected_object_overlap"
            frame_meta.append(entry)
            if idx % viz_stride == 0:
                _write_overlay(frame_rgb, bbox, None, f"{ep_path.name} frame {frame}: rejected hand bbox", viz_dir / f"hand_bbox_frame_{frame:06d}.png")
            continue

        np.save(bbox_path, bbox.astype(np.float32))
        detected += 1
        entry["bbox"] = bbox.astype(float).tolist()
        frame_meta.append(entry)
        if idx % viz_stride == 0:
            _write_overlay(frame_rgb, bbox, None, f"{ep_path.name} frame {frame}: hand bbox", viz_dir / f"hand_bbox_frame_{frame:06d}.png")

    meta = {
        "episode": ep_path.name,
        "stage": "hand_bbox",
        "frames_requested": [int(f) for f in frame_ids],
        "num_requested": int(len(frame_ids)),
        "num_detected": int(detected),
        "detection_ratio": float(detected / max(len(frame_ids), 1)),
        "prompts": prompts,
        "frames": frame_meta,
    }
    with (meta_dir / "hand_bbox_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[hand_bbox] {ep_path.name}: detected {detected}/{len(frame_ids)} frames")
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[hand_bbox] loading DINO on {device}")
    processor, dino_model = _load_dino(cfg, device)
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, processor, dino_model, device)
        except Exception as exc:
            print(f"[hand_bbox] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "hand_bbox_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[hand_bbox] failed {len(failed)} episodes; wrote {log_path}")
