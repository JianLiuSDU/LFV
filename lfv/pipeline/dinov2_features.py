from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def _load_model(cfg, device: str):
    hf_endpoint = _cfg_get(cfg, "runtime.hf_endpoint")
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))
        os.environ.setdefault("HF_HUB_ENDPOINT", str(hf_endpoint))

    model_id = str(_cfg_get(cfg, "dinov2.model_id", "facebook/dinov2-small"))
    local_weight_path = _cfg_get(cfg, "dinov2.local_weight_path")
    if local_weight_path:
        weight_path = Path(str(local_weight_path))
        if weight_path.exists():
            import timm

            timm_name = str(_cfg_get(cfg, "dinov2.timm_model_name", "vit_small_patch14_dinov2"))
            model = timm.create_model(timm_name, pretrained=False, dynamic_img_size=True)
            try:
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weight_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                raise RuntimeError(f"Missing keys while loading local DINOv2 weights: {missing[:8]}")
            if unexpected:
                print(f"[dinov2] warning: ignored unexpected local weight keys: {unexpected[:8]}")
            model = model.to(device)
            model._lfv_backend = "timm"
            model._lfv_weight_source = str(weight_path)
            model.eval()
            return None, model

    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        model._lfv_backend = "hf_transformers"
    except Exception as exc:
        print(f"[dinov2] warning: falling back to manual ImageNet preprocessing: {exc}")
        processor = None
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_id).to(device)
            model._lfv_backend = "hf_transformers"
        except Exception as model_exc:
            print(f"[dinov2] warning: falling back to timm DINOv2 model: {model_exc}")
            import timm

            timm_name = str(_cfg_get(cfg, "dinov2.timm_model_name", "vit_small_patch14_dinov2"))
            model = timm.create_model(timm_name, pretrained=True, dynamic_img_size=True).to(device)
            model._lfv_backend = "timm"
    model.eval()
    return processor, model


def _load_anchor_frame(ep_path: Path, cfg) -> int:
    timing_rel = str(_cfg_get(cfg, "dinov2.timing_path", "contact_timing/contact_timing.json"))
    timing_path = ep_path / timing_rel
    if timing_path.exists():
        with timing_path.open("r", encoding="utf-8") as f:
            timing = json.load(f)
        if timing.get("anchor_frame") is not None:
            return int(timing["anchor_frame"])
    return int(_cfg_get(cfg, "dinov2.anchor_frame", 0))


def _load_point_pixels(ep_path: Path, cfg) -> tuple[np.ndarray, str]:
    contact_rel = str(_cfg_get(cfg, "dinov2.contact_field_path", "contact_field/contact_field.npz"))
    contact_path = ep_path / contact_rel
    if contact_path.exists():
        data = np.load(contact_path)
        if "pixels_uv" in data:
            return data["pixels_uv"].astype(np.int64), str(contact_rel)

    sample_rel = str(_cfg_get(cfg, "dinov2.object_sample_path", "sample_points/sampled_2d_uniform.npy"))
    sample_path = ep_path / sample_rel
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing DINOv2 point pixel source: {sample_path}")
    sample = np.load(sample_path, allow_pickle=True)
    if hasattr(sample, "item"):
        sample = sample.item()
    if isinstance(sample, dict) and "query_points_2d" in sample:
        return np.asarray(sample["query_points_2d"], dtype=np.int64), str(sample_rel)
    return np.asarray(sample, dtype=np.int64), str(sample_rel)


def _extract_grid(frame_rgb: np.ndarray, processor, model, device: str) -> tuple[np.ndarray, dict]:
    backend = getattr(model, "_lfv_backend", "hf_transformers")
    h, w = frame_rgb.shape[:2]
    if processor is None or backend == "timm":
        img = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        pixel_values = ((img - mean) / std).unsqueeze(0)
        patch_hint = 14
        pad_h = (patch_hint - (h % patch_hint)) % patch_hint
        pad_w = (patch_hint - (w % patch_hint)) % patch_hint
        if pad_h or pad_w:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="replicate")
        pixel_values = pixel_values.to(device)
        inputs = {"pixel_values": pixel_values}
    else:
        image = Image.fromarray(frame_rgb)
        inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(device)
    with torch.no_grad():
        if backend == "timm":
            features = model.forward_features(inputs["pixel_values"])
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    tokens = features["x_norm_patchtokens"]
                elif "x" in features:
                    tokens = features["x"]
                else:
                    raise KeyError(f"Unsupported timm forward_features dict keys: {sorted(features)}")
            else:
                tokens = features
            prefix = int(getattr(model, "num_prefix_tokens", 1))
            if tokens.ndim != 3:
                raise ValueError(f"Expected timm tokens [B,N,C], got {tuple(tokens.shape)}")
            if tokens.shape[1] > prefix:
                tokens = tokens[:, prefix:, :]
        else:
            outputs = model(**inputs)
            tokens = outputs.last_hidden_state[:, 1:, :]
    spatial_h, spatial_w = int(inputs["pixel_values"].shape[-2]), int(inputs["pixel_values"].shape[-1])
    if backend == "timm":
        patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", 14)
        patch = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
    else:
        patch = int(getattr(model.config, "patch_size", 14))
    grid_h = spatial_h // patch
    grid_w = spatial_w // patch
    expected = grid_h * grid_w
    if tokens.shape[1] != expected:
        # Fall back to a rectangular factorization close to the image aspect ratio.
        n = int(tokens.shape[1])
        aspect = w / max(h, 1)
        grid_h = max(1, int(round((n / aspect) ** 0.5)))
        while grid_h > 1 and n % grid_h != 0:
            grid_h -= 1
        grid_w = n // grid_h
    feat = tokens.reshape(1, grid_h, grid_w, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()
    grid = feat[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    meta = {
        "image_height": int(h),
        "image_width": int(w),
        "padded_height": int(spatial_h),
        "padded_width": int(spatial_w),
        "patch_size": int(patch),
        "grid_height": int(grid_h),
        "grid_width": int(grid_w),
        "feature_dim": int(grid.shape[-1]),
        "backend": backend,
    }
    return grid, meta


def _sample_point_features(grid: np.ndarray, pixels_uv: np.ndarray, image_shape: tuple[int, int], device: str) -> np.ndarray:
    h, w = image_shape
    pixels = pixels_uv.astype(np.float32)
    if len(pixels) == 0:
        return np.zeros((0, grid.shape[-1]), dtype=np.float32)
    x = np.clip(pixels[:, 0], 0, max(w - 1, 0))
    y = np.clip(pixels[:, 1], 0, max(h - 1, 0))
    x_norm = (2.0 * x / max(w - 1, 1)) - 1.0
    y_norm = (2.0 * y / max(h - 1, 1)) - 1.0
    coords = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)
    feat = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0).to(device)
    sample_grid = torch.from_numpy(coords).view(1, len(coords), 1, 2).to(device)
    with torch.no_grad():
        sampled = F.grid_sample(feat, sample_grid, mode="bilinear", align_corners=True)
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
    return sampled


def process_episode(ep_path: str | Path, cfg, processor, model, device: str) -> bool:
    ep_path = Path(ep_path)
    out_dir = ep_path / str(_cfg_get(cfg, "dinov2.output_dir", "dinov2_features"))
    out_dir.mkdir(parents=True, exist_ok=True)
    point_file = out_dir / str(_cfg_get(cfg, "dinov2.point_feature_file", "point_dinov2_features.npy"))
    dense_file = out_dir / str(_cfg_get(cfg, "dinov2.dense_feature_file", "anchor_dinov2_grid.npz"))
    meta_file = out_dir / str(_cfg_get(cfg, "dinov2.meta_file", "dinov2_meta.json"))
    overwrite = bool(_cfg_get(cfg, "runtime.overwrite", False))
    if point_file.exists() and meta_file.exists() and not overwrite:
        print(f"[dinov2] skip existing {ep_path.name}")
        return True

    anchor_frame = _load_anchor_frame(ep_path, cfg)
    rgb = zarr.open(str(ep_path / "rgb"), mode="r")
    if anchor_frame < 0 or anchor_frame >= rgb.shape[0]:
        raise IndexError(f"anchor_frame {anchor_frame} outside video length {rgb.shape[0]}")
    frame_rgb = np.asarray(rgb[anchor_frame])
    pixels_uv, pixel_source = _load_point_pixels(ep_path, cfg)
    grid, grid_meta = _extract_grid(frame_rgb, processor, model, device)
    sample_shape = (int(grid_meta.get("padded_height", frame_rgb.shape[0])), int(grid_meta.get("padded_width", frame_rgb.shape[1])))
    point_features = _sample_point_features(grid, pixels_uv, sample_shape, device)

    np.save(point_file, point_features.astype(np.float32))
    if bool(_cfg_get(cfg, "dinov2.save_dense_grid", True)):
        np.savez_compressed(dense_file, features=grid.astype(np.float32))
    meta = {
        "episode": ep_path.name,
        "anchor_frame": int(anchor_frame),
        "pixel_source": pixel_source,
        "point_count": int(len(pixels_uv)),
        "point_feature_file": point_file.name,
        "dense_feature_file": dense_file.name if bool(_cfg_get(cfg, "dinov2.save_dense_grid", True)) else None,
        "model_id": str(_cfg_get(cfg, "dinov2.model_id", "facebook/dinov2-small")),
        **grid_meta,
    }
    np.save(out_dir / "point_pixels_uv.npy", pixels_uv.astype(np.int32))
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[dinov2] {ep_path.name}: points={len(pixels_uv)} dim={point_features.shape[-1]}")
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[dinov2] loading model on {device}: {_cfg_get(cfg, 'dinov2.model_id', 'facebook/dinov2-small')}")
    processor, model = _load_model(cfg, device)
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, processor, model, device)
        except Exception as exc:
            print(f"[dinov2] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        log_path = Path(cfg.paths.processed_root) / "dinov2_failed_logs.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[dinov2] failed {len(failed)} episodes; wrote {log_path}")
