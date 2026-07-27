from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import zarr

from lfv.data_processing.episode_io import iter_processed_episodes
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def _cfg_get(cfg, dotted_key: str, default=None):
    cur = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _device(cfg) -> str:
    requested = str(_cfg_get(cfg, "runtime.device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _add_sam2_to_path() -> None:
    root = Path(__file__).resolve().parents[2] / "third_party" / "sam2"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_dino(cfg, device: str):
    hf_endpoint = _cfg_get(cfg, "runtime.hf_endpoint")
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))
        os.environ.setdefault("HF_HUB_ENDPOINT", str(hf_endpoint))

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = str(_cfg_get(cfg, "hand.model_id", _cfg_get(cfg, "object.model_id", "IDEA-Research/grounding-dino-base")))
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def _load_sam2(cfg, device: str):
    _add_sam2_to_path()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(str(cfg.sam2.model_cfg), str(cfg.sam2.checkpoint), device=device)
    return SAM2ImagePredictor(model)


def _post_process_grounding(processor, outputs, inputs, image_size, box_threshold: float, text_threshold: float):
    post_process = processor.post_process_grounded_object_detection
    kwargs = {
        "outputs": outputs,
        "input_ids": inputs.input_ids,
        "text_threshold": text_threshold,
        "target_sizes": [image_size[::-1]],
    }
    signature = inspect.signature(post_process)
    if "box_threshold" in signature.parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    return post_process(**kwargs)[0]


def _detect_hand_bbox(frame_rgb: np.ndarray, prompts: list[str], cfg, processor, model, device: str) -> tuple[np.ndarray | None, dict]:
    image = Image.fromarray(frame_rgb)
    box_threshold = float(_cfg_get(cfg, "hand.box_threshold", 0.2))
    text_threshold = float(_cfg_get(cfg, "hand.text_threshold", 0.2))
    attempts = []
    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = _post_process_grounding(
            processor,
            outputs,
            inputs,
            image.size,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels", [])
        count = 0 if boxes is None else int(len(boxes))
        attempt = {"prompt": prompt, "num_boxes": count}
        if scores is not None and count:
            score_np = scores.detach().cpu().numpy()
            attempt["scores"] = score_np.astype(float).tolist()
        attempts.append(attempt)
        if count == 0:
            continue
        if scores is None:
            best_idx = 0
            best_score = None
        else:
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].detach().cpu().item())
        bbox = boxes[best_idx].detach().cpu().numpy().astype(np.float32)
        return bbox, {
            "status": "ok",
            "prompt": prompt,
            "score": best_score,
            "label": str(labels[best_idx]) if len(labels) > best_idx else "",
            "attempts": attempts,
        }
    return None, {"status": "missing", "attempts": attempts}


def _predict_mask(frame_rgb: np.ndarray, bbox: np.ndarray, predictor, device: str) -> tuple[np.ndarray, float]:
    autocast_enabled = device.startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        predictor.set_image(frame_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=True,
        )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(np.float32), float(scores[best_idx])


def _frame_indices(num_frames: int, cfg) -> list[int]:
    frames = _cfg_get(cfg, "hand.frames")
    if frames is not None:
        return sorted({int(f) for f in frames if 0 <= int(f) < num_frames})
    start = int(_cfg_get(cfg, "hand.frame_start", 0))
    end_cfg = _cfg_get(cfg, "hand.frame_end")
    end = num_frames - 1 if end_cfg is None else min(int(end_cfg), num_frames - 1)
    stride = max(1, int(_cfg_get(cfg, "hand.frame_stride", 3)))
    if end < start:
        return []
    return list(range(max(0, start), end + 1, stride))


def _write_overlay(frame_rgb: np.ndarray, bbox: np.ndarray | None, mask: np.ndarray | None, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_rgb)
    if mask is not None:
        ax.imshow(mask, cmap="autumn", alpha=(mask > 0.5).astype(np.float32) * 0.45)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="cyan", facecolor=(0, 0, 0, 0), lw=2)
        )
    ax.set_title(title)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def process_episode(ep_path: str | Path, cfg, processor, dino_model, sam_predictor, device: str) -> bool:
    ep_path = Path(ep_path)
    bbox_dir = ep_path / str(_cfg_get(cfg, "hand.bbox_dir", "hand_bbox"))
    mask_dir = ep_path / str(_cfg_get(cfg, "hand.mask_dir", "hand_mask"))
    meta_dir = ep_path / str(_cfg_get(cfg, "hand.meta_dir", "hand_contact"))
    viz_dir = ep_path / str(_cfg_get(cfg, "hand.viz_dir", "viz"))
    bbox_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    rgb = zarr.open(str(ep_path / "rgb"), mode="r")
    frame_ids = _frame_indices(int(rgb.shape[0]), cfg)
    if not frame_ids:
        raise ValueError(f"No hand frames selected for {ep_path.name}")

    prompts = [str(_cfg_get(cfg, "hand.prompt", "hand ."))]
    prompts.extend(str(p) for p in _cfg_get(cfg, "hand.fallback_prompts", []))
    overwrite = bool(_cfg_get(cfg, "runtime.overwrite", False))
    viz_stride = max(1, int(_cfg_get(cfg, "hand.viz_stride", 12)))

    frame_meta = []
    detected = 0
    for idx, frame in enumerate(frame_ids):
        bbox_path = bbox_dir / f"frame_{frame:06d}.npy"
        mask_path = mask_dir / f"frame_{frame:06d}.npy"
        if bbox_path.exists() and mask_path.exists() and not overwrite:
            frame_meta.append({"frame": int(frame), "status": "skipped_existing"})
            detected += 1
            continue

        frame_rgb = np.asarray(rgb[frame])
        bbox, det_meta = _detect_hand_bbox(frame_rgb, prompts, cfg, processor, dino_model, device)
        entry = {"frame": int(frame), **det_meta}
        if bbox is None:
            frame_meta.append(entry)
            if idx % viz_stride == 0:
                _write_overlay(frame_rgb, None, None, f"{ep_path.name} frame {frame}: no hand", viz_dir / f"hand_frame_{frame:06d}.png")
            continue

        mask, sam_score = _predict_mask(frame_rgb, bbox, sam_predictor, device)
        np.save(bbox_path, bbox.astype(np.float32))
        np.save(mask_path, mask.astype(np.float32))
        detected += 1
        entry.update(
            {
                "bbox": bbox.astype(float).tolist(),
                "sam_score": float(sam_score),
                "mask_pixels": int(np.sum(mask > 0.5)),
            }
        )
        frame_meta.append(entry)
        if idx % viz_stride == 0:
            _write_overlay(
                frame_rgb,
                bbox,
                mask,
                f"{ep_path.name} frame {frame}: hand score={entry.get('score')}",
                viz_dir / f"hand_frame_{frame:06d}.png",
            )

    meta = {
        "episode": ep_path.name,
        "frames_requested": [int(f) for f in frame_ids],
        "num_requested": int(len(frame_ids)),
        "num_detected": int(detected),
        "detection_ratio": float(detected / max(len(frame_ids), 1)),
        "prompts": prompts,
        "frames": frame_meta,
    }
    with (meta_dir / "hand_detection_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[hand] {ep_path.name}: detected {detected}/{len(frame_ids)} frames")
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[hand] loading DINO+SAM2 on {device}")
    processor, dino_model = _load_dino(cfg, device)
    sam_predictor = _load_sam2(cfg, device)

    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, processor, dino_model, sam_predictor, device)
        except Exception as exc:
            print(f"[hand] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "hand_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[hand] failed {len(failed)} episodes; wrote {log_path}")
