import numpy as np

from lfv.evaluation.functional_motion import (
    remove_endpoint_trend,
    trajectory_spectrum_summary,
)


def test_remove_endpoint_trend_annihilates_straight_bridge() -> None:
    progress = np.linspace(0.0, 1.0, 64)
    trajectory = np.stack((0.3 * progress, -0.1 * progress, progress), axis=-1)[None]
    assert np.allclose(remove_endpoint_trend(trajectory), 0.0, atol=1e-12)


def test_identical_mid_frequency_shape_has_unit_retention_and_phase() -> None:
    progress = np.linspace(0.0, 1.0, 64)
    trajectory = np.zeros((2, 64, 3), dtype=np.float64)
    trajectory[..., 0] = 0.2 * progress
    trajectory[..., 2] = 0.05 * np.sin(2.0 * np.pi * 8.0 * progress)
    metrics, _ = trajectory_spectrum_summary(trajectory, trajectory.copy())
    assert np.isclose(metrics["position_mid_energy_retention"], 1.0)
    assert np.isclose(metrics["position_mid_coefficient_cosine"], 1.0)
    assert np.isclose(metrics["detrended_shape_relative_l2"], 0.0)
    assert np.isclose(metrics["dominant_curvature_frame_error"], 0.0)
