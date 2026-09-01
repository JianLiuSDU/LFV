"""Build immutable Stage 2 episode artifacts from processed LFV data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from lfv.features import DinoV2DenseExtractor
from lfv.geometry.pose9d import camera_delta_to_local, matrix_to_pose9d_np
from lfv.utils.imagecodecs import register_image_codecs

from .audit import audit_dataset
from .sampling import (
    assert_unique_aligned,
    farthest_pixel_sample,
    unproject_pixels,
    valid_mask_pixels,
)
from .source_io import load_episode_calibration
from .splits import build_split_manifest


register_image_codecs()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_dino_grid(
    extractor: DinoV2DenseExtractor,
    rgb: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = rgb.shape[:2]
    patch = extractor.patch_size
    pad_h = (patch - height % patch) % patch
    pad_w = (patch - width % patch) % patch
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return extractor.extract(padded), padded.shape[:2]


def sample_dense_features(
    grid: np.ndarray,
    pixels_uv: np.ndarray,
    padded_shape: tuple[int, int],
) -> np.ndarray:
    height, width = padded_shape
    pixels = np.asarray(pixels_uv, dtype=np.float32)
    x = pixels[:, 0] * 2.0 / max(width - 1, 1) - 1.0
    y = pixels[:, 1] * 2.0 / max(height - 1, 1) - 1.0
    coordinates = torch.from_numpy(np.stack((x, y), axis=-1)).view(1, -1, 1, 2)
    features = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(
        features, coordinates, mode="bilinear", align_corners=True
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return F.normalize(sampled.float(), dim=-1).cpu().numpy().astype(np.float32)


def build_episode_cache(
    episode: str | Path,
    output_path: str | Path,
    extractor: DinoV2DenseExtractor,
    *,
    num_points: int = 256,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
    scene_scale: float = 1.0,
    object_instance_id: str = "",
) -> dict:
    episode = Path(episode)
    output = Path(output_path)
    depth_scale, intrinsic, _ = load_episode_calibration(episode)
    rgb = np.asarray(zarr.open(str(episode / "rgb"), mode="r")[0])
    depth = (
        np.asarray(zarr.open(str(episode / "depth"), mode="r")[0], dtype=np.float32)
        * depth_scale
    )
    manipulated_mask = np.load(episode / "sam_mask/affordance_mask.npy") > 0.5
    reference_mask = np.load(episode / "target_sam_mask/target_mask.npy") > 0.5
    manipulated_pixels = farthest_pixel_sample(
        valid_mask_pixels(manipulated_mask, depth, min_depth_m, max_depth_m),
        num_points,
    )
    reference_pixels = farthest_pixel_sample(
        valid_mask_pixels(reference_mask, depth, min_depth_m, max_depth_m),
        num_points,
    )
    manipulated_camera = unproject_pixels(manipulated_pixels, depth, intrinsic)
    reference_camera = unproject_pixels(reference_pixels, depth, intrinsic)
    scene_origin = manipulated_camera.mean(axis=0).astype(np.float32)
    manipulated_points = (
        (manipulated_camera - scene_origin) / float(scene_scale)
    ).astype(np.float32)
    reference_points = (
        (reference_camera - scene_origin) / float(scene_scale)
    ).astype(np.float32)

    grid, padded_shape = extract_dino_grid(extractor, rgb)
    manipulated_dino = sample_dense_features(grid, manipulated_pixels, padded_shape)
    reference_dino = sample_dense_features(grid, reference_pixels, padded_shape)
    assert_unique_aligned(
        manipulated_pixels, manipulated_points, manipulated_dino, num_points
    )
    assert_unique_aligned(reference_pixels, reference_points, reference_dino, num_points)

    trajectory_file = episode / "se3_trajectory/dp_action_trajectory.npz"
    trajectory_raw = np.load(trajectory_file)
    matrices_camera = np.asarray(trajectory_raw["T_matrices_4x4"], dtype=np.float32)
    matrices_local = np.stack(
        [
            camera_delta_to_local(matrix, scene_origin, scene_scale)
            for matrix in matrices_camera
        ]
    )
    trajectory_pose9d = matrix_to_pose9d_np(matrices_local)
    goal_pose9d = trajectory_pose9d[-1].copy()
    fingerprint = hashlib.sha256(
        (
            _sha256(episode / "sam_mask/affordance_mask.npy")
            + _sha256(episode / "target_sam_mask/target_mask.npy")
            + _sha256(trajectory_file)
        ).encode("ascii")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        manipulated_points=manipulated_points,
        manipulated_dino=manipulated_dino.astype(np.float16),
        manipulated_mask=np.ones(num_points, dtype=np.float32),
        manipulated_pixels_uv=manipulated_pixels.astype(np.int32),
        reference_points=reference_points,
        reference_dino=reference_dino.astype(np.float16),
        reference_mask=np.ones(num_points, dtype=np.float32),
        reference_pixels_uv=reference_pixels.astype(np.int32),
        goal_pose9d=goal_pose9d.astype(np.float32),
        trajectory_pose9d=trajectory_pose9d.astype(np.float32),
        scene_origin=scene_origin,
        scene_scale=np.asarray(scene_scale, dtype=np.float32),
        intrinsic=intrinsic,
        episode_id=np.asarray(episode.name),
        object_instance_id=np.asarray(object_instance_id),
        source_fingerprint=np.asarray(fingerprint),
    )
    return {
        "episode_id": episode.name,
        "object_instance_id": object_instance_id,
        "artifact": str(output.resolve()),
        "source_fingerprint": fingerprint,
        "dino_dim": int(manipulated_dino.shape[-1]),
        "num_points": int(num_points),
    }


def build_dataset_cache(
    source_root: str | Path,
    cache_root: str | Path,
    weights_path: str | Path,
    *,
    device: str = "cuda",
    num_points: int = 256,
    scene_scale: float = 1.0,
    overwrite: bool = False,
    instance_mapping_path: str | Path | None = None,
    allow_episode_id_as_instance: bool = True,
    limit: int | None = None,
) -> dict:
    source = Path(source_root)
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(source, num_points=num_points)
    (cache / "audit_report.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    accepted = [item for item in audit["episodes"] if item["accepted"]]
    if limit is not None:
        accepted = accepted[: int(limit)]
    mapping: dict[str, str] = {}
    if instance_mapping_path:
        mapping = json.loads(Path(instance_mapping_path).read_text(encoding="utf-8"))
    weights = Path(weights_path).expanduser().resolve()
    extractor = DinoV2DenseExtractor(
        model_name="vit_small_patch14_dinov2",
        weights_path=weights,
        device=device,
    )
    records = []
    for index, item in enumerate(accepted):
        episode = source / item["episode_id"]
        output = cache / "episodes" / f"{episode.name}.npz"
        if output.exists() and not overwrite:
            with np.load(output, allow_pickle=False) as data:
                record = {
                    "episode_id": str(data["episode_id"]),
                    "object_instance_id": str(data["object_instance_id"]),
                    "artifact": str(output.resolve()),
                    "source_fingerprint": str(data["source_fingerprint"]),
                    "dino_dim": int(data["manipulated_dino"].shape[-1]),
                    "num_points": int(data["manipulated_points"].shape[0]),
                }
        else:
            record = build_episode_cache(
                episode,
                output,
                extractor,
                num_points=num_points,
                scene_scale=scene_scale,
                object_instance_id=str(mapping.get(episode.name, "")),
            )
        records.append(record)
        print(f"[stage2-cache] {index + 1}/{len(accepted)} {episode.name}", flush=True)
    split = build_split_manifest(
        records,
        cache / "split_manifest.json",
        allow_episode_id_as_instance=allow_episode_id_as_instance,
    )
    manifest = {
        "version": 1,
        "source_root": str(source.resolve()),
        "cache_root": str(cache.resolve()),
        "num_points": num_points,
        "scene_scale": scene_scale,
        "dino": {
            "model": "vit_small_patch14_dinov2",
            "weights": str(weights),
            "weights_sha256": _sha256(weights),
            "feature_dim": records[0]["dino_dim"] if records else None,
        },
        "audit": {
            "episode_count": audit["episode_count"],
            "accepted_count": audit["accepted_count"],
            "rejected_count": audit["rejected_count"],
        },
        "split_quality": split["split_quality"],
        "records": records,
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
