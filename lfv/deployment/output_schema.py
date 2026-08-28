from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CameraPlanResult:
    """Serializable camera-frame handoff artifact for the execution computer."""

    selected_grasp_camera: np.ndarray | None = None
    object_trajectory_camera: np.ndarray | None = None
    tcp_trajectory_camera: np.ndarray | None = None
    gripper_commands: np.ndarray | None = None
    report: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.selected_grasp_camera is not None and np.asarray(self.selected_grasp_camera).shape != (4, 4):
            raise ValueError("selected_grasp_camera must be [4,4]")
        for name, value in (("object_trajectory_camera", self.object_trajectory_camera),
                            ("tcp_trajectory_camera", self.tcp_trajectory_camera)):
            if value is not None and np.asarray(value).ndim != 3:
                raise ValueError(f"{name} must be [T,4,4]")
        if self.object_trajectory_camera is not None and self.tcp_trajectory_camera is not None:
            if len(self.object_trajectory_camera) != len(self.tcp_trajectory_camera):
                raise ValueError("Object and TCP trajectories must have the same length")
