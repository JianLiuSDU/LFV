from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lfv.data_processing.episode_io import first_rgb_frame, iter_processed_episodes
from lfv.pipeline.object_specs import ObjectSpec, iter_object_specs


def _add_sam2_to_path() -> None:
    root = Path(__file__).resolve().parents[2] / "third_party" / "sam2"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _device(cfg) -> str:
    requested = str(cfg.runtime.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _show_mask_and_border(mask, ax, color=(1, 1, 0, 0.5)) -> None:
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        poly = cnt.reshape(-1, 2)
        if len(poly) > 2:
            ax.add_patch(plt.Polygon(poly, facecolor="none", edgecolor="yellow", linewidth=2))


def _show_box(box, ax) -> None:
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", facecolor=(0, 0, 0, 0), lw=1.5, linestyle="--")
    )


def _load_predictor(cfg, device: str):
    _add_sam2_to_path()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(str(cfg.sam2.model_cfg), str(cfg.sam2.checkpoint), device=device)
    return SAM2ImagePredictor(model)


def process_object(ep_path: Path, cfg, spec: ObjectSpec, predictor, device: str, initial_frame) -> bool:
    bbox_path = ep_path / spec.bbox_dir / spec.bbox_file
    if not bbox_path.exists():
        raise FileNotFoundError(f"Missing bbox file: {bbox_path}")

    mask_dir = ep_path / spec.mask_dir
    viz_dir = ep_path / "viz"
    mask_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / spec.mask_file
    if mask_path.exists() and not bool(cfg.runtime.overwrite):
        print(f"[sam2] skip existing {ep_path.name}/{spec.name}")
        return True

    input_box = np.load(bbox_path)

    autocast_enabled = device.startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        predictor.set_image(initial_frame)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    best_score = float(scores[best_idx])
    np.save(mask_path, best_mask)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(initial_frame)
    _show_mask_and_border(best_mask, ax)
    _show_box(input_box, ax)
    ax.set_title(f"{ep_path.name} | {spec.name} SAM2 score: {best_score:.4f}")
    ax.axis("off")
    fig.savefig(viz_dir / f"{spec.viz_prefix}_sam_overlay.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    Image.fromarray(best_mask.astype(np.uint8) * 255).save(viz_dir / f"{spec.viz_prefix}_sam_binary.png")
    if best_score < float(cfg.sam2.score_warn_threshold):
        print(f"[sam2] warn {ep_path.name}/{spec.name}: low score {best_score:.4f}")
    else:
        print(f"[sam2] {ep_path.name}/{spec.name}: score {best_score:.4f}")
    return True


def process_episode(ep_path: str | Path, cfg, specs: list[ObjectSpec], predictor, device: str) -> bool:
    ep_path = Path(ep_path)
    initial_frame = first_rgb_frame(ep_path)
    for spec in specs:
        process_object(ep_path, cfg, spec, predictor, device, initial_frame)
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[sam2] loading SAM2 on {device}")
    predictor = _load_predictor(cfg, device)
    specs = iter_object_specs(cfg)
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, specs, predictor, device)
        except Exception as exc:
            print(f"[sam2] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        print(f"[sam2] failed {len(failed)} episodes")
