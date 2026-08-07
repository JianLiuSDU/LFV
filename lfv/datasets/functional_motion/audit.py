"""Read-only audit of LFV pouring episode inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from lfv.utils.imagecodecs import register_image_codecs

from .sampling import valid_mask_pixels
from .source_io import load_episode_calibration


register_image_codecs()


REQUIRED = (
    "rgb",
    "depth",
    "sam_mask/affordance_mask.npy",
    "target_sam_mask/target_mask.npy",
    "se3_trajectory/dp_action_trajectory.npz",
)


def audit_episode(
    episode: str | Path,
    *,
    num_points: int = 256,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
) -> dict:
    episode = Path(episode)
    missing = [name for name in REQUIRED if not (episode / name).exists()]
    report = {"episode_id": episode.name, "accepted": False, "missing": missing}
    if missing:
        report["reason"] = "missing_required_files"
        return report
    try:
        depth_scale, _, calibration_source = load_episode_calibration(episode)
        rgb = zarr.open(str(episode / "rgb"), mode="r")
        depth_zarr = zarr.open(str(episode / "depth"), mode="r")
        if rgb.shape[0] == 0 or depth_zarr.shape[0] == 0:
            raise ValueError("empty RGB/depth array")
        depth = np.asarray(depth_zarr[0], dtype=np.float32) * depth_scale
        manipulated_mask = np.load(episode / "sam_mask/affordance_mask.npy") > 0.5
        reference_mask = np.load(episode / "target_sam_mask/target_mask.npy") > 0.5
        manipulated_candidates = valid_mask_pixels(
            manipulated_mask, depth, min_depth_m, max_depth_m
        )
        reference_candidates = valid_mask_pixels(
            reference_mask, depth, min_depth_m, max_depth_m
        )
        trajectory = np.load(episode / "se3_trajectory/dp_action_trajectory.npz")
        matrices = np.asarray(trajectory["T_matrices_4x4"], dtype=np.float32)
        rotations = matrices[:, :3, :3]
        rotation_error = float(
            np.max(np.abs(np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3)))
        )
        determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
        report.update(
            {
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth_zarr.shape),
                "calibration_source": calibration_source,
                "depth_scale": depth_scale,
                "manipulated_valid_candidates": int(len(manipulated_candidates)),
                "reference_valid_candidates": int(len(reference_candidates)),
                "trajectory_shape": list(matrices.shape),
                "trajectory_finite": bool(np.isfinite(matrices).all()),
                "start_identity_error": float(np.max(np.abs(matrices[0] - np.eye(4)))),
                "rotation_orthogonality_error": rotation_error,
                "rotation_determinant_error": determinant_error,
            }
        )
        report["accepted"] = bool(
            len(manipulated_candidates) >= num_points
            and len(reference_candidates) >= num_points
            and matrices.shape == (64, 4, 4)
            and report["trajectory_finite"]
            and report["start_identity_error"] < 1e-3
            and rotation_error < 1e-3
            and determinant_error < 1e-3
        )
        report["reason"] = "accepted" if report["accepted"] else "quality_gate_failed"
    except Exception as exc:
        report["reason"] = "exception"
        report["error"] = repr(exc)
    return report


def audit_dataset(source_root: str | Path, **kwargs) -> dict:
    root = Path(source_root)
    episodes = sorted(
        root.glob("episode_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    reports = [audit_episode(episode, **kwargs) for episode in episodes]
    return {
        "source_root": str(root.resolve()),
        "episode_count": len(reports),
        "accepted_count": sum(item["accepted"] for item in reports),
        "rejected_count": sum(not item["accepted"] for item in reports),
        "episodes": reports,
    }
