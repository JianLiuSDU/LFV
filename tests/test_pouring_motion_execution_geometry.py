import numpy as np
from scipy.spatial.transform import Rotation

from lfv.inference.functional_motion.two_stage_pouring import (
    local_pose7d_to_camera_delta,
    local_trajectory_to_world_object_poses,
    sample_heat_point_cloud,
    sample_mask_point_cloud,
)
from lfv.robot.panda_grasp_execution import (
    graspnet_object_row_to_panda_tcp_world,
    interpolate_se3,
    object_poses_to_tcp_poses,
    project_prismatic_trajectory,
    tcp_world_to_absolute_action,
)


def test_motion_point_sampling_contracts_are_exact_and_deterministic():
    height, width = 32, 40
    depth = np.ones((height, width), dtype=np.float32)
    intrinsic = np.asarray([[100, 0, 20], [0, 100, 16], [0, 0, 1]], dtype=np.float32)
    cup = np.zeros((height, width), dtype=bool)
    cup[8:24, 5:18] = True
    bowl = np.zeros_like(cup)
    bowl[10:27, 24:37] = True
    heat = np.zeros_like(depth)
    heat[8:24, 5:18] = np.linspace(0.1, 1.0, 16 * 13).reshape(16, 13)

    first = sample_heat_point_cloud(heat, cup, depth, intrinsic, 256)
    second = sample_heat_point_cloud(heat, cup, depth, intrinsic, 256)
    target_points, target_pixels = sample_mask_point_cloud(bowl, depth, intrinsic, 64)
    assert first[0].shape == (256, 3)
    assert first[1].shape == (256, 2)
    assert first[2].shape == (256,)
    assert target_points.shape == (64, 3)
    assert target_pixels.shape == (64, 2)
    np.testing.assert_array_equal(first[1], second[1])


def test_centroid_local_pose_maps_back_to_expected_camera_transform():
    centroid = np.asarray([0.2, -0.1, 0.7], dtype=np.float32)
    rotation = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    camera_translation = np.asarray([0.08, 0.04, -0.02], dtype=np.float32)
    local_translation = rotation @ centroid + camera_translation - centroid
    pose = np.concatenate(
        (local_translation, Rotation.from_matrix(rotation).as_quat())
    ).astype(np.float32)
    recovered = local_pose7d_to_camera_delta(pose, centroid)
    np.testing.assert_allclose(recovered[:3, :3], rotation, atol=1e-6)
    np.testing.assert_allclose(recovered[:3, 3], camera_translation, atol=1e-6)


def test_local_trajectory_world_mapping_and_rigid_grasp_attachment():
    identity_pose = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    translated_pose = np.asarray([0.1, 0.0, 0.2, 0, 0, 0, 1], dtype=np.float32)
    object_initial = np.eye(4, dtype=np.float32)
    object_initial[:3, 3] = [0.3, -0.2, 0.05]
    objects = local_trajectory_to_world_object_poses(
        np.stack((identity_pose, translated_pose)),
        np.asarray([0.0, 0.0, 1.0]),
        np.eye(4),
        object_initial,
    )
    np.testing.assert_allclose(objects[0], object_initial, atol=1e-6)
    np.testing.assert_allclose(
        objects[1, :3, 3], object_initial[:3, 3] + [0.1, 0.0, 0.2], atol=1e-6
    )
    tcp_initial = np.eye(4, dtype=np.float32)
    tcp_initial[:3, 3] = [0.31, -0.2, 0.15]
    tcp = object_poses_to_tcp_poses(objects, object_initial, tcp_initial)
    relative0 = np.linalg.inv(objects[0]) @ tcp[0]
    relative1 = np.linalg.inv(objects[1]) @ tcp[1]
    np.testing.assert_allclose(relative0, relative1, atol=1e-6)


def test_graspnet_axes_and_depth_convert_to_panda_tcp():
    row = np.zeros(17, dtype=np.float32)
    row[1] = 0.04
    row[3] = 0.03
    # GraspNet: X approach down, Y close in world Y, Z vertical in world X.
    row[4:13] = np.asarray(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32
    ).reshape(-1)
    row[13:16] = [0.2, 0.1, 0.3]
    tcp = graspnet_object_row_to_panda_tcp_world(row, np.eye(4))
    np.testing.assert_allclose(tcp[:3, 2], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(tcp[:3, 1], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(tcp[:3, 3], [0.2, 0.1, 0.27], atol=1e-6)
    assert np.linalg.det(tcp[:3, :3]) > 0.999


def test_se3_interpolation_excludes_start_and_hits_endpoint():
    start = np.eye(4, dtype=np.float32)
    end = np.eye(4, dtype=np.float32)
    end[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    end[:3, 3] = [0.2, -0.1, 0.3]
    poses = interpolate_se3(start, end, 5)
    assert poses.shape == (5, 4, 4)
    assert not np.allclose(poses[0], start)
    np.testing.assert_allclose(poses[-1], end, atol=1e-6)


def test_panda_action_keeps_absolute_arm_and_normalized_full_close_scalar():
    tcp_world = np.eye(4, dtype=np.float32)
    tcp_world[:3, 3] = [0.4, -0.2, 0.3]
    action = tcp_world_to_absolute_action(tcp_world, np.eye(4), -1.0)
    assert action.shape == (7,)
    np.testing.assert_allclose(action[:3], tcp_world[:3, 3], atol=1e-6)
    assert action[-1] == -1.0


def test_prismatic_projection_is_positive_monotonic_and_clamped():
    initial = np.eye(4, dtype=np.float32)
    poses = np.repeat(initial[None], 5, axis=0)
    poses[:, :3, 3] = np.asarray(
        [[0.0, 0.0, 0.0], [-0.03, 0.02, 0.01], [-0.02, 0.0, 0.0],
         [-0.12, -0.01, 0.02], [-0.10, 0.0, 0.0]], dtype=np.float32
    )
    projected, distance = project_prismatic_trajectory(
        poses, initial, np.asarray([-1.0, 0.0, 0.0]), max_displacement=0.08
    )
    np.testing.assert_allclose(distance, [0.0, 0.03, 0.03, 0.08, 0.08], atol=1e-6)
    np.testing.assert_allclose(projected[:, 0, 3], -distance, atol=1e-6)
    np.testing.assert_allclose(projected[:, 1:3, 3], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        projected[:, :3, :3], np.repeat(np.eye(3)[None], 5, axis=0), atol=1e-6
    )
