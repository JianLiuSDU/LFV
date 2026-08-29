#!/usr/bin/env python3
"""Execute an LFV camera plan on an Aubo arm.

The Aubo SDK is intentionally isolated in ``AuboAdapter``.  The default mode
is a dry-run that prints transformed poses; implement the five adapter methods
with the SDK used on the robot computer before passing ``--execute``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


class AuboAdapter:
    """Thin SDK boundary; replace method bodies with the local Aubo API."""

    def connect(self) -> None:
        raise NotImplementedError("Connect to Aubo in AuboAdapter.connect()")

    def move_linear(self, pose_base: np.ndarray, speed_m_s: float) -> None:
        raise NotImplementedError("Send a Cartesian pose in AuboAdapter.move_linear()")

    def open_gripper(self) -> None:
        raise NotImplementedError("Open the gripper in AuboAdapter.open_gripper()")

    def close_gripper(self) -> None:
        raise NotImplementedError("Close the gripper in AuboAdapter.close_gripper()")

    def stop(self) -> None:
        pass


def load_handeye(path: Path) -> tuple[np.ndarray, dict]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = np.asarray(cfg["T_base_camera"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"T_base_camera must be [4,4], got {matrix.shape}")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("T_base_camera must be homogeneous")
    return matrix, cfg


def _validate_pose_array(name: str, poses: np.ndarray, batched: bool = False) -> None:
    expected = (4, 4) if not batched else (poses.shape[0], 4, 4)
    if poses.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    rows = poses[..., 3, :]
    if not np.allclose(rows, [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{name} contains a non-homogeneous pose")


def transform_plan(plan_path: Path, handeye_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(plan_path, allow_pickle=False) as data:
        required = {"tcp_camera", "tcp_trajectory_camera"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"camera plan is missing required arrays: {missing}")
        grasp = np.asarray(data["tcp_camera"], dtype=np.float64)
        trajectory = np.asarray(data["tcp_trajectory_camera"], dtype=np.float64)
    _validate_pose_array("tcp_camera", grasp)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError(f"tcp_trajectory_camera must have shape [T,4,4], got {trajectory.shape}")
    _validate_pose_array("tcp_trajectory_camera", trajectory, batched=True)
    if trajectory.shape[0] == 0:
        raise ValueError("tcp_trajectory_camera must contain at least one step")
    if not np.allclose(trajectory[0], grasp, atol=1e-4):
        raise ValueError("trajectory[0] must equal tcp_camera; regenerate the plan with the corrected frame semantics")
    base_camera, cfg = load_handeye(handeye_path)
    base_grasp = base_camera @ grasp
    base_trajectory = base_camera[None] @ trajectory
    return {"grasp": base_grasp, "trajectory": base_trajectory}, cfg


def pose_summary(pose: np.ndarray) -> str:
    xyz = pose[:3, 3]
    # Quaternion avoids an Euler-angle gimbal-lock warning for top-down poses.
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return (
        f"xyz(m)=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f}), "
        f"quat_xyzw=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
    )


def run(plan: dict[str, np.ndarray], cfg: dict, robot: AuboAdapter | None, execute: bool) -> None:
    grasp = plan["grasp"]
    trajectory = plan["trajectory"]
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= pregrasp[:3, 2] * float(cfg.get("pregrasp_distance_m", 0.10))
    approach_steps = int(cfg.get("approach_steps", 20))
    alpha = np.linspace(0.0, 1.0, approach_steps + 1)[1:]
    approach = np.repeat(np.eye(4, dtype=np.float64)[None], len(alpha), axis=0)
    approach[:, :3, 3] = pregrasp[:3, 3][None] * (1.0 - alpha[:, None]) + grasp[:3, 3][None] * alpha[:, None]
    approach[:, :3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((pregrasp[:3, :3], grasp[:3, :3]))))(alpha).as_matrix()
    print("grasp:", pose_summary(grasp))
    print("trajectory:", len(trajectory), "steps; first:", pose_summary(trajectory[0]))
    if not execute:
        print("dry-run: no Aubo commands sent")
        return
    assert robot is not None
    robot.connect()
    try:
        robot.open_gripper()
        robot.move_linear(pregrasp, float(cfg.get("approach_speed_m_s", 0.02)))
        for pose in approach:
            robot.move_linear(pose, float(cfg.get("approach_speed_m_s", 0.02)))
        time.sleep(float(cfg.get("settle_seconds", 0.25)))
        robot.close_gripper()
        for pose in trajectory:
            robot.move_linear(pose, float(cfg.get("trajectory_speed_m_s", 0.05)))
    finally:
        robot.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--handeye", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan, cfg = transform_plan(args.plan, args.handeye)
    run(plan, cfg, AuboAdapter() if args.execute else None, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
