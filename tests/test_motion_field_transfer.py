import numpy as np

from lfv.deployment.rgbd_alignment import align_depth_to_rgb
from lfv.models.functional_motion_generation.motion_field_transfer import transport_motion_field


def test_rgbd_alignment_scales_depth_and_intrinsics():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.float32)
    k = np.array([[3.0, 0, 1.0], [0, 4.0, 0.5], [0, 0, 1]], dtype=np.float32)
    aligned, aligned_k, report = align_depth_to_rgb(rgb, depth, k)
    assert aligned.shape == (4, 6)
    np.testing.assert_allclose(aligned_k, [[6, 0, 2], [0, 8, 1], [0, 0, 1]])
    assert report["resized"] is True


def test_motion_field_transport_preserves_peak():
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    source_points = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1).astype(np.float32)
    target_points = source_points.copy()
    source_dino = np.stack((np.cos(angles), np.sin(angles), np.ones_like(angles)), axis=1).astype(np.float32)
    target_dino = source_dino.copy()
    source_field = np.exp(-((angles - 0.3) ** 2) / 0.08).astype(np.float32)
    result = transport_motion_field(source_points, source_dino, source_field, target_points, target_dino, node_count=32, graph_neighbors=6, graph_maximum_neighbors=12)
    assert result.target_field.shape == (32,)
    assert np.isclose(result.target_field.sum(), 1.0, atol=1e-5)
    assert int(result.target_field.argmax()) == int(source_field.argmax())
