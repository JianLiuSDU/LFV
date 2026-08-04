from __future__ import annotations

import numpy as np

from lfv.geometry import UpperHandleOracleConfig, build_upper_handle_oracle_heat


def test_upper_handle_oracle_excludes_body_and_lower_handle():
    rng = np.random.default_rng(4)
    angles = rng.uniform(-np.pi, np.pi, 4000)
    body = np.stack(
        (
            0.035 * np.cos(angles),
            0.035 * np.sin(angles),
            rng.uniform(-0.03, 0.03, len(angles)),
        ),
        axis=-1,
    )
    upper_handle = rng.normal(
        loc=[0.045, -0.045, 0.018],
        scale=[0.003, 0.003, 0.004],
        size=(200, 3),
    )
    lower_handle = rng.normal(
        loc=[0.045, -0.045, -0.025],
        scale=[0.003, 0.003, 0.003],
        size=(200, 3),
    )
    points = np.concatenate((body, upper_handle, lower_handle)).astype(np.float32)
    result = build_upper_handle_oracle_heat(
        points,
        config=UpperHandleOracleConfig(
            protrusion_min_m=0.045,
            z_min_m=-0.002,
            z_max_m=0.030,
        ),
    )
    body_heat = result.heat[: len(body)]
    upper_heat = result.heat[len(body) : len(body) + len(upper_handle)]
    lower_heat = result.heat[-len(lower_handle) :]
    assert upper_heat.max() > 0.9
    assert np.quantile(upper_heat, 0.5) > 0.1
    assert body_heat.max() == 0.0
    assert lower_heat.max() == 0.0
