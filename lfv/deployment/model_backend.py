from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotionPrediction:
    goal_camera: np.ndarray
    object_trajectory_camera: np.ndarray
    metadata: dict[str, Any]


class ExternalMotionBackend:
    """Adapter for an arbitrary model runner writing ``motion_prediction.npz``."""

    def __init__(self, command: str, output_name: str = "motion_prediction.npz"):
        self.command, self.output_name = command, output_name

    def predict(self, *, workdir: str | Path, **_: Any) -> MotionPrediction:
        workdir = Path(workdir)
        subprocess.run(self.command.format(workdir=shlex.quote(str(workdir))), shell=True, check=True, cwd=str(workdir))
        payload = np.load(workdir / self.output_name, allow_pickle=False)
        goal = np.asarray(payload["goal_camera"] if "goal_camera" in payload else payload["goal_pose_camera"], dtype=np.float32).reshape(4, 4)
        trajectory = np.asarray(payload["object_trajectory_camera"], dtype=np.float32).reshape(-1, 4, 4)
        return MotionPrediction(goal, trajectory, {"backend": "external", "artifact": str(workdir / self.output_name)})


class LegacyPouringBackend:
    """Run the existing two-checkpoint pouring adapter in camera coordinates.

    The historical script emits world-frame poses.  The deployment wrapper
    intentionally sets world==camera in its temporary snapshot, so the output
    contract remains camera-frame and the second computer only needs its
    calibrated hand-eye transform.
    """

    def __init__(self, *, model_repo: str, goal_checkpoint: str, trajectory_checkpoint: str, language_embedding: str | None = None, python_executable: str = "python", steps_script: str | None = None, seed: int = 42, device: str = "cuda:0"):
        self.model_repo = Path(model_repo).expanduser()
        self.goal_checkpoint = str(Path(goal_checkpoint).expanduser())
        self.trajectory_checkpoint = str(Path(trajectory_checkpoint).expanduser())
        self.language_embedding = language_embedding
        self.python_executable = python_executable
        self.steps_script = Path(steps_script).expanduser() if steps_script else Path(__file__).resolve().parents[2] / "scripts" / "inference" / "infer_pouring_motion.py"
        self.seed, self.device = int(seed), device

    def predict(self, *, workdir: str | Path, rgb: np.ndarray, depth_m: np.ndarray, cup_mask: np.ndarray, bowl_mask: np.ndarray, intrinsic_cv: np.ndarray, heatmap: np.ndarray) -> MotionPrediction:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        snapshot = workdir / "model_snapshot.npz"
        transfer = workdir / "model_transfer.npz"
        np.savez_compressed(snapshot, rgb=np.asarray(rgb, dtype=np.uint8), depth_m=np.asarray(depth_m, dtype=np.float32), cup_mask=np.asarray(cup_mask, dtype=bool), bowl_mask=np.asarray(bowl_mask, dtype=bool), intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32), T_world_to_camera=np.eye(4, dtype=np.float32), T_object_to_world=np.eye(4, dtype=np.float32))
        np.savez_compressed(transfer, target_heatmap=np.asarray(heatmap, dtype=np.float32))
        output_dir = workdir / "legacy_motion"
        command = [self.python_executable, str(self.steps_script), "--snapshot", str(snapshot), "--transfer-result", str(transfer), "--output-dir", str(output_dir), "--model-repo", str(self.model_repo), "--goal-checkpoint", self.goal_checkpoint, "--trajectory-checkpoint", self.trajectory_checkpoint, "--seed", str(self.seed), "--device", self.device]
        if self.language_embedding:
            command += ["--language-embedding", str(Path(self.language_embedding).expanduser())]
        subprocess.run(command, check=True, cwd=str(self.model_repo))
        payload = np.load(output_dir / "pouring_motion_prediction.npz", allow_pickle=False)
        from lfv.inference.functional_motion.two_stage_pouring import local_pose7d_to_camera_delta
        goal = local_pose7d_to_camera_delta(payload["goal_pose7d_local_stage2"], payload["manipulated_centroid_camera_stage2"])
        trajectory = np.asarray(payload["pred_object_poses_world"], dtype=np.float32)
        return MotionPrediction(goal.astype(np.float32), trajectory, {"backend": "legacy_pouring", "legacy_output": str(output_dir)})
