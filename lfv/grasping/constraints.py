from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CollisionLimits:
    global_iou: float = 0.02
    finger_iou: float = 0.02
    palm_iou: float = 0.01
    approach_path_iou: float = 0.01

    def as_dict(self) -> dict[str, float]:
        return {
            "global_iou": self.global_iou,
            "left_finger_iou": self.finger_iou,
            "right_finger_iou": self.finger_iou,
            "palm_iou": self.palm_iou,
            "approach_path_iou": self.approach_path_iou,
        }


def strict_collision_mask(
    detector_collision: np.ndarray,
    part_ious: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    limits: CollisionLimits,
) -> np.ndarray:
    """Require both the GraspNet detector and every gripper part to pass."""
    collision = np.asarray(detector_collision, dtype=bool).reshape(-1)
    arrays = tuple(np.asarray(values, dtype=np.float64).reshape(-1) for values in part_ious)
    if len(arrays) != 5 or any(len(values) != len(collision) for values in arrays):
        raise ValueError("Collision outputs must contain five equally-sized part arrays")
    if any(value < 0 for value in limits.as_dict().values()):
        raise ValueError("Collision limits must be non-negative")
    return (
        (~collision)
        & (arrays[0] <= limits.global_iou)
        & (arrays[1] <= limits.finger_iou)
        & (arrays[2] <= limits.finger_iou)
        & (arrays[3] <= limits.palm_iou)
        & (arrays[4] <= limits.approach_path_iou)
    )
