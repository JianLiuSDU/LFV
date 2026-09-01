import numpy as np

from lfv.evaluation.functional_motion import pouring_success


def _ring(radius=0.05, height=0.0, count=32):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, height)),
        axis=-1,
    ).astype(np.float32)


def test_rim_over_opening_uses_rim_not_cup_center():
    result = pouring_success(
        _ring(),
        np.zeros(3, dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        0.06,
        min_rof=0.20,
    )
    assert result["rim_over_opening_fraction"] == 1.0
    assert result["continuous_rim_arc_fraction"] == 1.0
    assert result["success"]


def test_partial_rim_and_height_are_reflected_in_success():
    rim = _ring()
    rim[:16, 0] += 0.10
    rim[16:, 2] = 0.05
    result = pouring_success(
        rim,
        np.zeros(3, dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        0.06,
        min_rof=0.20,
        height_tolerance_m=0.01,
    )
    assert 0.0 < result["rim_over_opening_fraction"] < 0.20
    assert not result["success"]
