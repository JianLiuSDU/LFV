from __future__ import annotations

import os

# 在导入 transformers 之前设置 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lfv.data_processing.episode_io import first_rgb_frame, iter_processed_episodes
from lfv.pipeline.object_specs import ObjectSpec, iter_object_specs


def _device(cfg) -> str:
    requested = str(cfg.runtime.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _show_box(box, ax) -> None:
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def _load_model(cfg, device):
    hf_endpoint = cfg.runtime.get("hf_endpoint")
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))
        os.environ.setdefault("HF_HUB_ENDPOINT", str(hf_endpoint))

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.object.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.object.model_id).to(device)
    return processor, model


def get_object_bbox(initial_frame, text: str, device: str, processor, model, box_threshold: float, text_threshold: float):
    image = Image.fromarray(initial_frame)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    post_process = processor.post_process_grounded_object_detection
    kwargs = {
        "outputs": outputs,
        "input_ids": inputs.input_ids,
        "text_threshold": text_threshold,
        "target_sizes": [image.size[::-1]],
    }
    signature = inspect.signature(post_process)
    if "box_threshold" in signature.parameters:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    results = post_process(**kwargs)
    if len(results[0]["boxes"]) == 0:
        raise ValueError(f"No object detected for text prompt: {text!r}")
    return results[0]["boxes"].detach().cpu().numpy()[0]


def process_object(ep_path: Path, cfg, spec: ObjectSpec, processor, model, device: str, initial_frame) -> bool:
    bbox_dir = ep_path / spec.bbox_dir
    viz_dir = ep_path / "viz"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    bbox_path = bbox_dir / spec.bbox_file
    if bbox_path.exists() and not bool(cfg.runtime.overwrite):
        print(f"[dino] skip existing {ep_path.name}/{spec.name}")
        return True

    bbox = get_object_bbox(
        initial_frame,
        spec.prompt,
        device,
        processor,
        model,
        float(cfg.object.box_threshold),
        float(cfg.object.text_threshold),
    )
    np.save(bbox_path, bbox)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(initial_frame)
    _show_box(bbox, ax)
    ax.axis("off")
    ax.set_title(f"{ep_path.name} | {spec.name}: {spec.prompt}")
    fig.savefig(viz_dir / f"{spec.viz_prefix}_dino_detection.png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    print(f"[dino] {ep_path.name}/{spec.name}: {bbox.astype(int)}")
    return True


def process_episode(ep_path: str | Path, cfg, specs: list[ObjectSpec], processor, model, device: str) -> bool:
    ep_path = Path(ep_path)
    initial_frame = first_rgb_frame(ep_path)
    for spec in specs:
        process_object(ep_path, cfg, spec, processor, model, device, initial_frame)
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[dino] loading {cfg.object.model_id} on {device}")
    processor, model = _load_model(cfg, device)
    specs = iter_object_specs(cfg)
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, specs, processor, model, device)
        except Exception as exc:
            print(f"[dino] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "dino_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[dino] failed {len(failed)} episodes; wrote {log_path}")
