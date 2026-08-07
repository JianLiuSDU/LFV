"""Temporal-spectrum diagnostics for Stage 2 translation trajectories."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.ndimage import gaussian_filter1d


SPECTRAL_BANDS: dict[str, tuple[int, int]] = {
    "low": (1, 5),
    "mid": (5, 17),
    "high": (17, 33),
}


def remove_endpoint_trend(translation: np.ndarray) -> np.ndarray:
    """Remove each trajectory's straight start-to-end bridge.

    This separates the task-relevant path shape from the endpoint displacement.
    The input shape is ``[..., T, 3]`` and the returned shape is unchanged.
    """

    values = np.asarray(translation, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(f"Expected [..., T, 3], received {values.shape}")
    progress = np.linspace(0.0, 1.0, values.shape[-2], dtype=np.float64)
    progress = progress.reshape((1,) * (values.ndim - 2) + (-1, 1))
    bridge = values[..., :1, :] + progress * (
        values[..., -1:, :] - values[..., :1, :]
    )
    return values - bridge


def temporal_dct(signal: np.ndarray) -> np.ndarray:
    """Return an orthonormal DCT-II over the trajectory time dimension."""

    return dct(np.asarray(signal, dtype=np.float64), axis=-2, norm="ortho")


def _paired_cosine(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.reshape(prediction.shape[0], -1)
    gt = target.reshape(target.shape[0], -1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(gt, axis=1)
    valid = denominator > 1e-12
    if not np.any(valid):
        return 1.0 if np.allclose(pred, gt, atol=1e-10) else 0.0
    cosine = np.sum(pred[valid] * gt[valid], axis=1) / denominator[valid]
    return float(np.mean(cosine))


def _dominant_curvature_frame(translation: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter1d(
        np.asarray(translation, dtype=np.float64), sigma=1.0, axis=-2, mode="nearest"
    )
    acceleration = np.diff(smoothed, n=2, axis=-2)
    magnitude = np.linalg.norm(acceleration, axis=-1)
    if magnitude.shape[-1] > 4:
        magnitude[..., :2] = -np.inf
        magnitude[..., -2:] = -np.inf
    # Acceleration index i is centered at original frame i + 1.
    return np.argmax(magnitude, axis=-1) + 1


def trajectory_spectrum_summary(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Measure endpoint-independent frequency and phase fidelity.

    Args:
        prediction: Translation samples shaped ``[N, T, 3]`` in metres.
        target: Ground-truth translations with the same shape.
    """

    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected matching [N, T, 3], got {pred.shape} and {gt.shape}")

    pred_shape = remove_endpoint_trend(pred)
    gt_shape = remove_endpoint_trend(gt)
    pred_coeff = temporal_dct(pred_shape)
    gt_coeff = temporal_dct(gt_shape)
    pred_velocity_coeff = temporal_dct(np.diff(pred, axis=1))
    gt_velocity_coeff = temporal_dct(np.diff(gt, axis=1))

    metrics: dict[str, float] = {
        "translation_mse_m2": float(np.mean((pred - gt) ** 2)),
        "endpoint_translation_error_m": float(
            np.mean(np.linalg.norm(pred[:, -1] - gt[:, -1], axis=-1))
        ),
        "first_step_prediction_m": float(
            np.mean(np.linalg.norm(pred[:, 1] - pred[:, 0], axis=-1))
        ),
        "first_step_target_m": float(
            np.mean(np.linalg.norm(gt[:, 1] - gt[:, 0], axis=-1))
        ),
    }
    pred_length = np.linalg.norm(np.diff(pred, axis=1), axis=-1).sum(axis=-1)
    gt_length = np.linalg.norm(np.diff(gt, axis=1), axis=-1).sum(axis=-1)
    metrics["path_length_ratio"] = float(
        np.mean(pred_length / np.maximum(gt_length, 1e-12))
    )
    shape_denominator = np.linalg.norm(gt_coeff.reshape(gt.shape[0], -1), axis=-1)
    shape_error = np.linalg.norm(
        (pred_coeff - gt_coeff).reshape(gt.shape[0], -1), axis=-1
    )
    metrics["detrended_shape_relative_l2"] = float(
        np.mean(shape_error / np.maximum(shape_denominator, 1e-12))
    )
    pred_turn = _dominant_curvature_frame(pred)
    gt_turn = _dominant_curvature_frame(gt)
    metrics["dominant_curvature_frame_error"] = float(
        np.mean(np.abs(pred_turn - gt_turn))
    )

    for name, (start, stop) in SPECTRAL_BANDS.items():
        stop = min(stop, pred_coeff.shape[1])
        pred_band = pred_coeff[:, start:stop]
        gt_band = gt_coeff[:, start:stop]
        pred_energy = float(np.mean(np.sum(pred_band**2, axis=(1, 2))))
        gt_energy = float(np.mean(np.sum(gt_band**2, axis=(1, 2))))
        metrics[f"position_{name}_energy_pred"] = pred_energy
        metrics[f"position_{name}_energy_gt"] = gt_energy
        metrics[f"position_{name}_energy_retention"] = pred_energy / max(
            gt_energy, 1e-12
        )
        metrics[f"position_{name}_coefficient_cosine"] = _paired_cosine(
            pred_band, gt_band
        )

        velocity_stop = min(stop, pred_velocity_coeff.shape[1])
        pred_velocity_band = pred_velocity_coeff[:, start:velocity_stop]
        gt_velocity_band = gt_velocity_coeff[:, start:velocity_stop]
        pred_velocity_energy = float(
            np.mean(np.sum(pred_velocity_band**2, axis=(1, 2)))
        )
        gt_velocity_energy = float(
            np.mean(np.sum(gt_velocity_band**2, axis=(1, 2)))
        )
        metrics[f"velocity_{name}_energy_retention"] = (
            pred_velocity_energy / max(gt_velocity_energy, 1e-12)
        )
        metrics[f"velocity_{name}_coefficient_cosine"] = _paired_cosine(
            pred_velocity_band, gt_velocity_band
        )

    spectra = {
        "position_frequency": np.arange(pred_coeff.shape[1]),
        "position_magnitude_prediction": np.mean(
            np.linalg.norm(pred_coeff, axis=-1), axis=0
        ),
        "position_magnitude_target": np.mean(
            np.linalg.norm(gt_coeff, axis=-1), axis=0
        ),
        "velocity_frequency": np.arange(pred_velocity_coeff.shape[1]),
        "velocity_magnitude_prediction": np.mean(
            np.linalg.norm(pred_velocity_coeff, axis=-1), axis=0
        ),
        "velocity_magnitude_target": np.mean(
            np.linalg.norm(gt_velocity_coeff, axis=-1), axis=0
        ),
        "dominant_curvature_frame_prediction": pred_turn,
        "dominant_curvature_frame_target": gt_turn,
    }
    return metrics, spectra
