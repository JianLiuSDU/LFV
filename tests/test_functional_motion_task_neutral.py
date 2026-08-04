import numpy as np

from lfv.inference.functional_motion.two_stage import (
    local_pose7d_to_camera_delta,
    local_trajectory_to_world_object_poses,
    localize_clouds,
)


def test_task_neutral_motion_adapter_identity_and_translation():
    manipulated = np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32)
    reference = np.asarray([[0.0, 0.2, 1.0], [0.2, 0.2, 1.0]], dtype=np.float32)
    man_local, ref_local, centroid = localize_clouds(manipulated, reference)
    np.testing.assert_allclose(man_local.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(ref_local, reference - centroid, atol=1e-7)

    trajectory = np.asarray(
        [[0, 0, 0, 0, 0, 0, 1], [0.1, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    delta = local_pose7d_to_camera_delta(trajectory[1], centroid)
    np.testing.assert_allclose(delta[:3, 3], [0.1, 0, 0], atol=1e-7)
    world = local_trajectory_to_world_object_poses(
        trajectory, centroid, np.eye(4), np.eye(4)
    )
    np.testing.assert_allclose(world[0], np.eye(4), atol=1e-7)
    np.testing.assert_allclose(world[1, :3, 3], [0.1, 0, 0], atol=1e-7)
