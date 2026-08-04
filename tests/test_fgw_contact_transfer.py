from __future__ import annotations

import numpy as np

from lfv.affordance_transfer.fgw_contact_transfer import (
    farthest_point_indices,
    interpolate_node_heat,
    lift_part_to_points,
    normalized_knn_geodesic,
    solve_fgw,
)
from lfv.affordance_transfer.preprocessing import prepare_image
from lfv.affordance_transfer.schema import RGBDPart


def test_lift_part_uses_aligned_pixels_features_and_heat() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    heat = np.zeros((8, 8), dtype=np.float32)
    heat[3:5, 3:5] = 0.7
    prepared = prepare_image(
        rgb, mask, heatmap=heat, input_size=8, patch_size=2, bbox_margin=0.0
    )
    yy, xx = np.indices((4, 4))
    grid = np.stack((xx, yy, np.ones_like(xx)), axis=-1).astype(np.float32)
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)
    rgbd = RGBDPart(
        depth_m=np.ones((8, 8), dtype=np.float32),
        intrinsic_cv=np.asarray([[4, 0, 4], [0, 4, 4], [0, 0, 1]], np.float32),
        part_mask=mask,
    )
    cloud = lift_part_to_points(
        rgbd, grid, prepared.transform, heatmap=heat, maximum_depth_m=2.0
    )
    assert cloud.points_camera.shape == (16, 3)
    assert cloud.features.shape == (16, 3)
    assert cloud.pixels_uv.shape == (16, 2)
    assert cloud.heat is not None
    assert np.isclose(cloud.heat.max(), 0.7)
    assert np.allclose(np.linalg.norm(cloud.features, axis=1), 1.0)


def test_fps_is_seeded_and_does_not_duplicate_points() -> None:
    points = np.stack((np.arange(20), np.zeros(20), np.zeros(20)), axis=-1)
    first = farthest_point_indices(points, 8, seed=4)
    second = farthest_point_indices(points, 8, seed=4)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 8


def test_geodesic_is_invariant_to_uniform_scale() -> None:
    x = np.linspace(0, 1, 24)
    points = np.stack((x, 0.02 * np.sin(4 * x), np.zeros_like(x)), axis=-1)
    distance_a, _ = normalized_knn_geodesic(
        points, neighbors=3, maximum_neighbors=7, edge_length_ratio=4.0
    )
    distance_b, _ = normalized_knn_geodesic(
        3.7 * points, neighbors=3, maximum_neighbors=7, edge_length_ratio=4.0
    )
    assert np.allclose(distance_a, distance_b, atol=1e-6)


def test_fgw_transports_a_contact_field_without_minmax_rescaling() -> None:
    count = 12
    x = np.linspace(0, 1, count)
    structure = np.abs(x[:, None] - x[None, :])
    # Unique orthogonal descriptors make the intended structural orientation
    # unambiguous while still exercising the fused semantic/geometry objective.
    features = np.eye(count, dtype=np.float32)
    heat = (0.62 * np.exp(-((x - 0.5) ** 2) / 0.025)).astype(np.float32)
    result = solve_fgw(
        features,
        features,
        structure,
        structure,
        heat,
        alpha=0.5,
        maximum_iterations=100,
    )
    assert result.transport.shape == (count, count)
    assert np.allclose(result.transport.sum(axis=1), 1.0 / count, atol=1e-5)
    assert np.allclose(result.transport.sum(axis=0), 1.0 / count, atol=1e-5)
    assert int(np.argmax(result.target_node_heat)) in (5, 6)
    # The source field peak is 0.62-ish, so a hidden min-max normalization to
    # one would violate this assertion.
    assert result.target_node_heat.max() < 0.7
    assert np.allclose(result.target_node_heat, heat, atol=1e-3)


def test_node_heat_interpolation_preserves_constant_probability() -> None:
    nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    queries = np.asarray([[0.2, 0.2, 0], [0.8, 0.1, 0]], dtype=np.float32)
    output = interpolate_node_heat(
        nodes, np.full(3, 0.37, dtype=np.float32), queries, neighbors=3
    )
    assert np.allclose(output, 0.37)
