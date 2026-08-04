import numpy as np

from lfv.grasping import CollisionLimits, strict_collision_mask


def test_strict_collision_mask_checks_every_gripper_part():
    detector_collision = np.array([False, False, False, True])
    part_ious = (
        np.array([0.0, 0.03, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.03, 0.0]),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
    )
    keep = strict_collision_mask(
        detector_collision,
        part_ious,
        CollisionLimits(global_iou=0.02, finger_iou=0.02),
    )
    np.testing.assert_array_equal(keep, [True, False, False, False])


def test_strict_collision_mask_rejects_malformed_detector_output():
    arrays = tuple(np.zeros(2) for _ in range(5))
    try:
        strict_collision_mask(np.zeros(3, dtype=bool), arrays, CollisionLimits())
    except ValueError as exc:
        assert "equally-sized" in str(exc)
    else:
        raise AssertionError("Expected collision shape validation")
