from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class UpperHandleOracleConfig:
    """YCB mug upper-handle oracle in the object coordinate frame.

    ``handle_axis_xy`` points from the mug body toward the handle.  The
    protrusion threshold removes the cylindrical body before a compact
    Gaussian is evaluated on the upper handle surface.
    """

    handle_axis_xy: tuple[float, float] = (1.0, -1.0)
    protrusion_min_m: float = 0.045
    z_min_m: float = -0.002
    z_max_m: float = 0.030
    center_protrusion_m: float = 0.057
    center_z_m: float = 0.018
    sigma_protrusion_m: float = 0.010
    sigma_lateral_m: float = 0.012
    sigma_z_m: float = 0.009
    min_candidate_points: int = 32


@dataclass
class UpperHandleOracleResult:
    heat: np.ndarray
    candidate_mask: np.ndarray
    body_center_xy: np.ndarray
    handle_axis_xy: np.ndarray
    lateral_axis_xy: np.ndarray
    gaussian_center_object: np.ndarray

    def summary(self) -> dict:
        positive = self.heat > 0
        return {
            "num_points": int(len(self.heat)),
            "num_handle_candidates": int(self.candidate_mask.sum()),
            "num_heat_points": int(positive.sum()),
            "num_heat_points_above_0_1": int((self.heat > 0.1).sum()),
            "num_heat_points_above_0_5": int((self.heat > 0.5).sum()),
            "heat_max": float(self.heat.max(initial=0.0)),
            "body_center_xy": self.body_center_xy.astype(float).tolist(),
            "handle_axis_xy": self.handle_axis_xy.astype(float).tolist(),
            "gaussian_center_object": self.gaussian_center_object.astype(float).tolist(),
        }


def build_upper_handle_oracle_heat(
    points_object: np.ndarray,
    *,
    config: UpperHandleOracleConfig | None = None,
) -> UpperHandleOracleResult:
    config = config or UpperHandleOracleConfig()
    points = np.asarray(points_object, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_object must have shape [N,3], got {points.shape}")
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("points_object must be non-empty and finite")

    handle_axis = np.asarray(config.handle_axis_xy, dtype=np.float32)
    axis_norm = float(np.linalg.norm(handle_axis))
    if axis_norm < 1e-8:
        raise ValueError("handle_axis_xy cannot be zero")
    handle_axis /= axis_norm
    lateral_axis = np.asarray(
        [handle_axis[1], -handle_axis[0]],
        dtype=np.float32,
    )
    body_center = np.median(points[:, :2], axis=0).astype(np.float32)
    centered_xy = points[:, :2] - body_center[None]
    protrusion = centered_xy @ handle_axis
    lateral = centered_xy @ lateral_axis
    candidate_mask = (
        (protrusion >= config.protrusion_min_m)
        & (points[:, 2] >= config.z_min_m)
        & (points[:, 2] <= config.z_max_m)
    )
    if int(candidate_mask.sum()) < config.min_candidate_points:
        raise ValueError(
            "Too few upper-handle points after geometric filtering: "
            f"{int(candidate_mask.sum())} < {config.min_candidate_points}"
        )

    lateral_center = float(np.median(lateral[candidate_mask]))
    gaussian = np.exp(
        -0.5
        * (
            np.square(
                (protrusion - config.center_protrusion_m)
                / max(config.sigma_protrusion_m, 1e-8)
            )
            + np.square(
                (lateral - lateral_center)
                / max(config.sigma_lateral_m, 1e-8)
            )
            + np.square(
                (points[:, 2] - config.center_z_m)
                / max(config.sigma_z_m, 1e-8)
            )
        )
    ).astype(np.float32)
    gaussian[~candidate_mask] = 0.0
    maximum = float(gaussian.max(initial=0.0))
    if maximum > 0:
        gaussian /= maximum

    center_xy = (
        body_center
        + config.center_protrusion_m * handle_axis
        + lateral_center * lateral_axis
    )
    gaussian_center = np.asarray(
        [center_xy[0], center_xy[1], config.center_z_m],
        dtype=np.float32,
    )
    return UpperHandleOracleResult(
        heat=gaussian,
        candidate_mask=candidate_mask,
        body_center_xy=body_center,
        handle_axis_xy=handle_axis,
        lateral_axis_xy=lateral_axis,
        gaussian_center_object=gaussian_center,
    )


def upper_handle_oracle_config_dict(
    config: UpperHandleOracleConfig,
) -> dict:
    return asdict(config)
