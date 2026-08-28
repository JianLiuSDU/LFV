from __future__ import annotations

import numpy as np


def trajectory_report(trajectory_camera: np.ndarray, *, min_depth_m: float = 1e-4, max_step_m: float = 0.15) -> dict[str, float | int | bool]:
    poses = np.asarray(trajectory_camera, dtype=np.float32).reshape(-1, 4, 4)
    if len(poses) == 0:
        raise ValueError("Trajectory is empty")
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) if len(poses) > 1 else np.zeros(0)
    valid = bool(np.isfinite(poses).all() and np.all(poses[:, 2, 3] > min_depth_m))
    return {"steps": int(len(poses)), "valid": valid, "max_translation_step_m": float(steps.max()) if len(steps) else 0.0, "mean_translation_step_m": float(steps.mean()) if len(steps) else 0.0, "excessive_step": bool(len(steps) and steps.max() > max_step_m)}


def completion_report(visible_points: np.ndarray, complete_points: np.ndarray, rms_m: float | None = None) -> dict[str, float | int | bool]:
    visible = np.asarray(visible_points)
    complete = np.asarray(complete_points)
    return {"visible_points": int(len(visible)), "complete_points": int(len(complete)), "completion_ratio": float(len(complete) / max(len(visible), 1)), "registration_rms_m": None if rms_m is None else float(rms_m), "usable": bool(len(complete) >= len(visible) and len(complete) >= 3)}
