"""Pure geometry helpers for executing GraspNet poses with a Panda TCP."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def maniskill_wxyz_pose_to_matrix(raw_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(raw_pose, dtype=np.float32).reshape(7)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(
        pose[[4, 5, 6, 3]]
    ).as_matrix().astype(np.float32)
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_maniskill_wxyz_pose(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32)
    return np.concatenate((transform[:3, 3], xyzw[[3, 0, 1, 2]])).astype(np.float32)


def graspnet_object_row_to_panda_tcp_world(
    grasp_row_object: np.ndarray,
    object_to_world: np.ndarray,
) -> np.ndarray:
    """Convert a GraspNet object-frame row to ``panda_hand_tcp`` in world."""

    row = np.asarray(grasp_row_object, dtype=np.float32).reshape(17)
    object_to_world = np.asarray(object_to_world, dtype=np.float32).reshape(4, 4)
    rotation_gn_world = object_to_world[:3, :3] @ row[4:13].reshape(3, 3)
    origin_world = object_to_world[:3, :3] @ row[13:16] + object_to_world[:3, 3]
    approaching = rotation_gn_world[:, 0]
    approaching /= max(float(np.linalg.norm(approaching)), 1e-8)
    closing = rotation_gn_world[:, 1]
    closing -= approaching * float(closing @ approaching)
    closing /= max(float(np.linalg.norm(closing)), 1e-8)
    orthogonal = np.cross(closing, approaching)
    orthogonal /= max(float(np.linalg.norm(orthogonal)), 1e-8)
    contact_center = origin_world + approaching * float(row[3])
    tcp = np.eye(4, dtype=np.float32)
    tcp[:3, :3] = np.stack((orthogonal, closing, approaching), axis=-1)
    tcp[:3, 3] = contact_center
    return tcp


def pregrasp_pose(tcp_grasp_world: np.ndarray, retreat_distance: float) -> np.ndarray:
    pose = np.asarray(tcp_grasp_world, dtype=np.float32).copy()
    pose[:3, 3] -= pose[:3, 2] * float(retreat_distance)
    return pose


def interpolate_se3(start: np.ndarray, end: np.ndarray, steps: int) -> np.ndarray:
    """Return ``steps`` poses including the endpoint and excluding the start."""

    if steps < 1:
        raise ValueError("steps must be positive")
    start = np.asarray(start, dtype=np.float32).reshape(4, 4)
    end = np.asarray(end, dtype=np.float32).reshape(4, 4)
    alpha = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[1:]
    translation = (
        start[:3, 3][None] * (1.0 - alpha[:, None])
        + end[:3, 3][None] * alpha[:, None]
    )
    rotations = Rotation.from_matrix(np.stack((start[:3, :3], end[:3, :3])))
    rotation = Slerp([0.0, 1.0], rotations)(alpha).as_matrix()
    output = np.repeat(np.eye(4, dtype=np.float32)[None], steps, axis=0)
    output[:, :3, :3] = rotation.astype(np.float32)
    output[:, :3, 3] = translation.astype(np.float32)
    return output


def object_poses_to_tcp_poses(
    object_poses_world: np.ndarray,
    object_to_world_initial: np.ndarray,
    tcp_grasp_world: np.ndarray,
) -> np.ndarray:
    object_to_tcp = (
        np.linalg.inv(np.asarray(object_to_world_initial, dtype=np.float32))
        @ np.asarray(tcp_grasp_world, dtype=np.float32)
    )
    return (
        np.asarray(object_poses_world, dtype=np.float32)
        @ object_to_tcp[None]
    ).astype(np.float32)


def project_prismatic_trajectory(
    poses: np.ndarray,
    initial: np.ndarray,
    axis: np.ndarray,
    max_displacement: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project poses onto one monotonic positive prismatic-joint axis."""

    output = np.asarray(poses, dtype=np.float32).copy()
    initial = np.asarray(initial, dtype=np.float32).reshape(4, 4)
    axis = np.asarray(axis, dtype=np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    scalars = (output[:, :3, 3] - initial[:3, 3]) @ axis
    scalars = np.maximum.accumulate(np.maximum(scalars, 0.0))
    if max_displacement is not None:
        scalars = np.minimum(scalars, float(max_displacement))
    output[:, :3, 3] = initial[:3, 3] + scalars[:, None] * axis
    output[:, :3, :3] = initial[:3, :3]
    return output, scalars


def tcp_world_to_absolute_action(
    tcp_world: np.ndarray,
    robot_root_world: np.ndarray,
    gripper_action: float,
) -> np.ndarray:
    """Build the mixed Panda action: absolute EE pose plus normalized gripper.

    ManiSkill's Panda ``pd_ee_pose`` composite controller leaves the arm pose
    unnormalized but normalizes the mimic-gripper scalar. Therefore ``+1`` is
    fully open and ``-1`` is the lower-limit/full-close command.
    """

    tcp_base = np.linalg.inv(robot_root_world) @ np.asarray(tcp_world, dtype=np.float32)
    euler_xyz = Rotation.from_matrix(tcp_base[:3, :3]).as_euler("XYZ")
    return np.concatenate(
        (tcp_base[:3, 3], euler_xyz, [float(gripper_action)])
    ).astype(np.float32)
