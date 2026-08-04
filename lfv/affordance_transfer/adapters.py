from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from lfv.utils.imagecodecs import register_image_codecs

from .schema import RGBDPart, SourceContactExample, TargetObservation


def load_source_episode(
    episode_dir: str | Path,
    *,
    frame_index: int,
    mask_path: str = "sam_mask/affordance_mask.npy",
    heatmap_path: str = "contact_heatmap/contact_heatmap.npz",
    heatmap_key: str = "heatmap_2d",
) -> SourceContactExample:
    """Load only the source RGB, object mask, and continuous contact heat."""
    import zarr

    episode_dir = Path(episode_dir).expanduser().resolve()
    register_image_codecs()
    rgb_array = zarr.open(str(episode_dir / "rgb"), mode="r")
    if not 0 <= frame_index < len(rgb_array):
        raise IndexError(f"frame_index={frame_index} is outside [0,{len(rgb_array)}).")
    rgb = np.asarray(rgb_array[frame_index]).copy()
    mask = np.load(episode_dir / mask_path).astype(bool)
    with np.load(episode_dir / heatmap_path) as archive:
        if heatmap_key not in archive:
            raise KeyError(f"{heatmap_key!r} not found in {episode_dir / heatmap_path}.")
        heatmap = np.asarray(archive[heatmap_key], dtype=np.float32).copy()
    return SourceContactExample(
        rgb=rgb,
        mask=mask,
        heatmap=heatmap,
        sample_id=f"{episode_dir.name}:frame_{frame_index:06d}",
    )


def load_target_snapshot(
    snapshot_path: str | Path,
    *,
    rgb_key: str = "rgb",
    mask_key: str = "cup_mask",
) -> TargetObservation:
    """Load a simulation RGB/mask pair.

    Depth, camera intrinsics, point clouds, and grasp fields in the NPZ are
    intentionally not read by this first-stage adapter.
    """
    snapshot_path = Path(snapshot_path).expanduser().resolve()
    with np.load(snapshot_path) as archive:
        rgb = np.asarray(archive[rgb_key]).copy()
        mask = np.asarray(archive[mask_key]).astype(bool).copy()
    return TargetObservation(rgb=rgb, mask=mask, sample_id=snapshot_path.stem)


def _intrinsic_from_realsense_metadata(metadata: dict, key: str) -> np.ndarray:
    values = metadata[key]
    return np.asarray(
        [
            [values["fx"], 0.0, values["ppx"]],
            [0.0, values["fy"], values["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_source_rgbd_part(
    episode_dir: str | Path,
    *,
    frame_index: int,
    part_mask_path: str,
    metadata_path: str = "meta.json",
    intrinsics_key: str = "color_intrinsics",
) -> RGBDPart:
    """Load an RGB-aligned RealSense depth frame for FGW geometry."""

    import zarr

    episode_dir = Path(episode_dir).expanduser().resolve()
    metadata = json.loads((episode_dir / metadata_path).read_text(encoding="utf-8"))
    depth_array = zarr.open(str(episode_dir / "depth"), mode="r")
    if not 0 <= frame_index < len(depth_array):
        raise IndexError(f"frame_index={frame_index} is outside [0,{len(depth_array)}).")
    depth_m = np.asarray(depth_array[frame_index], dtype=np.float32)
    depth_m *= float(metadata["depth_scale"])
    part_mask = np.load(episode_dir / part_mask_path).astype(bool)
    return RGBDPart(
        depth_m=depth_m,
        intrinsic_cv=_intrinsic_from_realsense_metadata(metadata, intrinsics_key),
        part_mask=part_mask,
    )


def load_target_rgbd_part(
    snapshot_path: str | Path,
    *,
    depth_key: str = "depth_m",
    intrinsics_key: str = "intrinsic_cv",
    part_mask_key: str = "cup_mask",
) -> RGBDPart:
    """Load aligned target RGB-D and the complete visible functional-part mask."""

    snapshot_path = Path(snapshot_path).expanduser().resolve()
    with np.load(snapshot_path) as archive:
        return RGBDPart(
            depth_m=np.asarray(archive[depth_key], dtype=np.float32).copy(),
            intrinsic_cv=np.asarray(archive[intrinsics_key], dtype=np.float32).copy(),
            part_mask=np.asarray(archive[part_mask_key]).astype(bool).copy(),
        )
