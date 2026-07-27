from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.pipeline.hand_segmentation import (
    _cfg_get,
    _device,
    _frame_indices,
    _load_sam2,
    _predict_mask,
    _write_overlay,
)
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def process_episode(ep_path: str | Path, cfg, predictor, device: str) -> bool:
    ep_path = Path(ep_path)
    bbox_dir = ep_path / str(_cfg_get(cfg, "hand.bbox_dir", "hand_bbox"))
    mask_dir = ep_path / str(_cfg_get(cfg, "hand.mask_dir", "hand_mask"))
    meta_dir = ep_path / str(_cfg_get(cfg, "hand.meta_dir", "hand_contact"))
    viz_dir = ep_path / str(_cfg_get(cfg, "hand.viz_dir", "viz"))
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    rgb = zarr.open(str(ep_path / "rgb"), mode="r")
    frame_ids = _frame_indices(int(rgb.shape[0]), cfg)
    overwrite = bool(_cfg_get(cfg, "runtime.overwrite", False))
    viz_stride = max(1, int(_cfg_get(cfg, "hand.viz_stride", 12)))

    frame_meta = []
    segmented = 0
    missing_bbox = 0
    for idx, frame in enumerate(frame_ids):
        bbox_path = bbox_dir / f"frame_{frame:06d}.npy"
        mask_path = mask_dir / f"frame_{frame:06d}.npy"
        if mask_path.exists() and not overwrite:
            segmented += 1
            frame_meta.append({"frame": int(frame), "status": "skipped_existing"})
            continue
        if not bbox_path.exists():
            if overwrite and mask_path.exists():
                mask_path.unlink()
            missing_bbox += 1
            frame_meta.append({"frame": int(frame), "status": "missing_bbox"})
            continue

        frame_rgb = np.asarray(rgb[frame])
        bbox = np.load(bbox_path).astype(np.float32)
        mask, sam_score = _predict_mask(frame_rgb, bbox, predictor, device)
        np.save(mask_path, mask.astype(np.float32))
        segmented += 1
        entry = {
            "frame": int(frame),
            "status": "ok",
            "bbox": bbox.astype(float).tolist(),
            "sam_score": float(sam_score),
            "mask_pixels": int(np.sum(mask > 0.5)),
        }
        frame_meta.append(entry)
        if idx % viz_stride == 0:
            _write_overlay(frame_rgb, bbox, mask, f"{ep_path.name} frame {frame}: hand mask", viz_dir / f"hand_mask_frame_{frame:06d}.png")

    meta = {
        "episode": ep_path.name,
        "stage": "hand_mask",
        "frames_requested": [int(f) for f in frame_ids],
        "num_requested": int(len(frame_ids)),
        "num_segmented": int(segmented),
        "num_missing_bbox": int(missing_bbox),
        "segmentation_ratio": float(segmented / max(len(frame_ids), 1)),
        "frames": frame_meta,
    }
    with (meta_dir / "hand_mask_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[hand_mask] {ep_path.name}: segmented {segmented}/{len(frame_ids)} frames; missing_bbox={missing_bbox}")
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[hand_mask] loading SAM2 on {device}")
    predictor = _load_sam2(cfg, device)
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, predictor, device)
        except Exception as exc:
            print(f"[hand_mask] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "hand_mask_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[hand_mask] failed {len(failed)} episodes; wrote {log_path}")
