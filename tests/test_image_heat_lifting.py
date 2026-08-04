import numpy as np

from lfv.lifting import lift_image_heat_to_camera


def test_lift_image_heat_uses_aligned_pixel_depth_and_intrinsic():
    heat = np.zeros((4, 5), dtype=np.float32)
    heat[1, 3] = 0.8
    heat[2, 2] = 0.1
    depth = np.ones((4, 5), dtype=np.float32) * 2.0
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 3] = True
    mask[2, 2] = True
    intrinsic = np.array(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    lifted = lift_image_heat_to_camera(
        heat,
        depth,
        mask,
        intrinsic,
        heat_threshold=0.2,
        maximum_depth_m=2.1,
    )
    np.testing.assert_array_equal(lifted.pixels_uv, [[3, 1], [2, 2]])
    np.testing.assert_allclose(lifted.points_camera[0], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(lifted.points_camera[1], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(lifted.raw_heat, [0.8, 0.1])
    np.testing.assert_allclose(lifted.heat, [0.8, 0.0])


def test_lift_rejects_misaligned_inputs_and_empty_threshold():
    heat = np.zeros((3, 3), dtype=np.float32)
    depth = np.ones((3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=bool)
    intrinsic = np.eye(3, dtype=np.float32)
    try:
        lift_image_heat_to_camera(heat, depth, mask, intrinsic, heat_threshold=0.5)
    except ValueError as exc:
        assert "exceeds heat_threshold" in str(exc)
    else:
        raise AssertionError("Expected empty heat rejection")

    try:
        lift_image_heat_to_camera(heat, depth[:2], mask, intrinsic)
    except ValueError as exc:
        assert "spatially aligned" in str(exc)
    else:
        raise AssertionError("Expected spatial alignment rejection")
