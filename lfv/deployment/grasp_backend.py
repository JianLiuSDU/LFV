from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lfv.robot.panda_grasp_execution import graspnet_object_row_to_panda_tcp_world


@dataclass(frozen=True)
class GraspSelection:
    tcp_camera: np.ndarray
    grasp_row: np.ndarray
    score: float
    metadata: dict[str, Any]


def _row_to_tcp_camera(row: np.ndarray, camera_from_object: np.ndarray) -> np.ndarray:
    return graspnet_object_row_to_panda_tcp_world(row, camera_from_object)


class NPZGraspBackend:
    """Select a GraspNet candidate from an offline ``[M,17]`` artifact."""

    def __init__(self, path: str | Path, *, preferred_approach_camera: tuple[float, float, float] = (0.0, -1.0, 0.0), topdown_weight: float = 0.5):
        self.path = Path(path).expanduser()
        self.preferred = np.asarray(preferred_approach_camera, dtype=np.float32)
        self.preferred /= max(float(np.linalg.norm(self.preferred)), 1e-8)
        self.topdown_weight = float(topdown_weight)

    def select(self, points_camera: np.ndarray, heat_camera: np.ndarray, camera_from_object: np.ndarray, *, workdir: str | Path | None = None) -> GraspSelection:
        payload = np.load(self.path, allow_pickle=False)
        rows = np.asarray(payload["grasps"] if "grasps" in payload else payload[payload.files[0]], dtype=np.float32)
        rows = rows.reshape(-1, 17)
        if not len(rows):
            raise ValueError("GraspNet artifact contains no candidates")
        points = np.asarray(points_camera, dtype=np.float32)
        heat = np.asarray(heat_camera, dtype=np.float32).reshape(-1)
        if len(points) != len(heat):
            raise ValueError("points_camera and heat_camera length mismatch")
        values = []
        for row in rows:
            tcp = _row_to_tcp_camera(row, camera_from_object)
            nearest = int(np.argmin(np.sum((points - tcp[:3, 3]) ** 2, axis=1)))
            heat_score = float(heat[nearest])
            approach = tcp[:3, 2] / max(float(np.linalg.norm(tcp[:3, 2])), 1e-8)
            topdown = max(0.0, float(approach @ self.preferred))
            detector_score = float(row[0])
            values.append(detector_score + heat_score + self.topdown_weight * topdown)
        index = int(np.argmax(values))
        tcp = _row_to_tcp_camera(rows[index], camera_from_object)
        return GraspSelection(tcp, rows[index], float(values[index]), {"candidate_index": index, "candidate_count": len(rows), "topdown_preferred_camera": self.preferred.tolist()})


class ExternalGraspNetBackend:
    """Run a GraspNet wrapper that writes ``grasps.npz`` then select a candidate."""

    def __init__(self, command: str, output_name: str = "grasps.npz", **kwargs: Any):
        self.command = command
        self.output_name = output_name
        self.kwargs = kwargs

    def select(self, points_camera: np.ndarray, heat_camera: np.ndarray, camera_from_object: np.ndarray, *, workdir: str | Path) -> GraspSelection:
        workdir = Path(workdir)
        command = self.command.format(workdir=shlex.quote(str(workdir)))
        subprocess.run(command, shell=True, check=True, cwd=str(workdir))
        artifact = workdir / self.output_name
        if not artifact.exists():
            raise FileNotFoundError(f"GraspNet command did not produce {artifact}")
        return NPZGraspBackend(artifact, **self.kwargs).select(points_camera, heat_camera, camera_from_object)
