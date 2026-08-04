from __future__ import annotations

import numpy as np

from lfv.geometry import (
    ContactHeatPropagationConfig,
    propagate_contact_heat_to_opposite_surface,
)


def _parallel_surfaces() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_x, grid_z = np.meshgrid(
        np.linspace(-0.010, 0.010, 17),
        np.linspace(-0.008, 0.008, 15),
    )
    left = np.stack(
        (grid_x.reshape(-1), np.full(grid_x.size, -0.006), grid_z.reshape(-1)),
        axis=-1,
    )
    right = left.copy()
    right[:, 1] = 0.006
    points = np.concatenate((left, right)).astype(np.float32)
    normals = np.zeros_like(points)
    normals[: len(left), 1] = -1.0
    normals[len(left) :, 1] = 1.0
    visible_points = left[np.linalg.norm(left[:, [0, 2]], axis=-1) < 0.006]
    visible_heat = np.exp(
        -np.square(np.linalg.norm(visible_points[:, [0, 2]], axis=-1) / 0.004)
    ).astype(np.float32)
    return visible_points, visible_heat, points, normals


def test_heat_propagates_only_to_antipodal_surface():
    visible_points, visible_heat, full_points, normals = _parallel_surfaces()
    result = propagate_contact_heat_to_opposite_surface(
        visible_points,
        visible_heat,
        full_points,
        normals,
        config=ContactHeatPropagationConfig(
            projection_radius=0.002,
            min_contact_width=0.010,
            max_contact_width=0.014,
            min_antipodal_cos=0.9,
            min_normal_opposition_cos=0.9,
            hidden_distance=0.004,
            min_pair_score=0.01,
        ),
    )
    midpoint = len(full_points) // 2
    assert result.visible_heat[:midpoint].max() > 0.9
    assert result.opposite_heat[midpoint:].max() > 0.8
    assert result.opposite_heat[:midpoint].max() == 0.0
    assert result.pairs
    assert all(pair.opposite_is_hidden for pair in result.pairs)
    assert all(0.012 <= pair.width_m <= 0.014 for pair in result.pairs)
    assert any(abs(pair.width_m - 0.012) < 1e-5 for pair in result.pairs)


def test_no_opposite_heat_when_normals_are_not_antipodal():
    visible_points, visible_heat, full_points, normals = _parallel_surfaces()
    normals[len(full_points) // 2 :] = np.array([0.0, -1.0, 0.0], np.float32)
    result = propagate_contact_heat_to_opposite_surface(
        visible_points,
        visible_heat,
        full_points,
        normals,
        config=ContactHeatPropagationConfig(
            projection_radius=0.002,
            min_contact_width=0.010,
            max_contact_width=0.014,
            min_antipodal_cos=0.9,
            min_normal_opposition_cos=0.9,
            min_pair_score=0.01,
        ),
    )
    assert not result.pairs
    assert result.opposite_heat.max() == 0.0
