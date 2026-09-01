"""Source-canonical field memory and target-to-source alignment utilities.

The functions in this module are deliberately agnostic to how the
source-to-target correspondence was obtained.  DINO/FGW/cycle code can
produce a row-normalized correspondence matrix; V7 then uses that matrix to
pull target observations onto the source canonical support.  No target
demonstration and no hand-defined functional coordinate are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def row_normalize_correspondence(
    correspondence: np.ndarray,
    *,
    target_mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return a finite row-stochastic source-to-target correspondence."""

    matrix = np.asarray(correspondence, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"correspondence must be [N_source,N_target], got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("correspondence must be finite and non-negative")
    matrix = matrix.copy()
    if target_mask is not None:
        valid = np.asarray(target_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"target mask length {valid.shape[0]} != correspondence target {matrix.shape[1]}"
            )
        matrix[:, ~valid] = 0.0
    row_sum = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sum <= eps):
        raise ValueError("correspondence contains an empty source row")
    return matrix / row_sum


def pull_target_to_source(
    correspondence: np.ndarray,
    target_points: np.ndarray,
    target_dino: np.ndarray,
    *,
    target_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull target XYZ/DINO observations onto source canonical rows.

    Returns ``(aligned_points, aligned_dino, row_confidence)``.  Each output
    row is a barycentric target observation and therefore has the source
    canonical support indexing required by V7's local encoder.
    """

    points = np.asarray(target_points, dtype=np.float32)
    dino = np.asarray(target_dino, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"target_points must be [N,3], got {points.shape}")
    if dino.ndim != 2 or dino.shape[0] != points.shape[0]:
        raise ValueError(f"target_dino must align with points, got {dino.shape}")
    matrix = row_normalize_correspondence(
        correspondence, target_mask=target_mask
    )
    if matrix.shape[1] != points.shape[0]:
        raise ValueError(
            f"correspondence target dimension {matrix.shape[1]} != target points {points.shape[0]}"
        )
    aligned_points = matrix @ points
    aligned_dino = matrix @ dino
    aligned_dino /= np.linalg.norm(aligned_dino, axis=1, keepdims=True).clip(min=1e-8)
    # Peak probability is an intentionally simple, calibrated localization
    # confidence.  The caller may multiply it by cycle confidence.
    confidence = matrix.max(axis=1).astype(np.float32)
    return aligned_points.astype(np.float32), aligned_dino.astype(np.float32), confidence


def canonical_field_gate(
    canonical_field: np.ndarray,
    row_confidence: np.ndarray,
    *,
    field_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Combine a source canonical field with correspondence confidence."""

    field = np.asarray(canonical_field, dtype=np.float32).reshape(-1)
    confidence = np.asarray(row_confidence, dtype=np.float32).reshape(-1)
    if field.shape != confidence.shape:
        raise ValueError(f"field/confidence shapes differ: {field.shape} vs {confidence.shape}")
    output = np.clip(field, 0.0, 1.0) * np.clip(confidence, 0.0, 1.0)
    if field_mask is not None:
        mask = np.asarray(field_mask, dtype=np.float32).reshape(-1)
        if mask.shape != output.shape:
            raise ValueError(f"field mask shape {mask.shape} != field {output.shape}")
        output *= np.clip(mask, 0.0, 1.0)
    return output.astype(np.float32)


def map_field_to_canonical(
    episode_field: np.ndarray,
    episode_to_canonical: np.ndarray,
    *,
    episode_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one episode field to canonical rows using an explicit matrix.

    ``episode_to_canonical`` has shape ``[N_canonical,N_episode]`` and its
    rows are normalized internally.  A zero row is rejected instead of being
    silently replaced by a fake correspondence.
    """

    field = np.asarray(episode_field, dtype=np.float32).reshape(-1)
    mapping = row_normalize_correspondence(
        episode_to_canonical, target_mask=episode_mask
    )
    if mapping.shape[1] != field.shape[0]:
        raise ValueError(
            f"mapping episode dimension {mapping.shape[1]} != field length {field.shape[0]}"
        )
    mapped = mapping @ np.clip(field, 0.0, None)
    coverage = mapping.max(axis=1)
    return mapped.astype(np.float32), coverage.astype(np.float32)


def aggregate_canonical_fields(
    fields: Sequence[np.ndarray],
    episode_to_canonical: Sequence[np.ndarray],
    *,
    sample_weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build canonical mean, variance and coverage confidence.

    Every episode must provide an explicit canonical mapping.  This function
    intentionally fails if one is missing; episode identity alone is not a
    geometric alignment.
    """

    if len(fields) == 0 or len(fields) != len(episode_to_canonical):
        raise ValueError("fields and canonical mappings must be non-empty and have equal length")
    mapped_fields: list[np.ndarray] = []
    coverages: list[np.ndarray] = []
    for field, mapping in zip(fields, episode_to_canonical):
        mapped, coverage = map_field_to_canonical(field, mapping)
        mapped_fields.append(mapped)
        coverages.append(coverage)
    length = mapped_fields[0].shape[0]
    if any(value.shape[0] != length for value in mapped_fields):
        raise ValueError("all canonical mappings must have the same row count")
    if sample_weights is None:
        weights = np.ones(len(mapped_fields), dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != len(mapped_fields) or (weights < 0).any():
            raise ValueError("sample_weights must be non-negative and match field count")
    if float(weights.sum()) <= 1e-12:
        raise ValueError("sample_weights sum to zero")
    weights = weights / weights.sum()
    stack = np.stack(mapped_fields, axis=0).astype(np.float64)
    mean = np.sum(stack * weights[:, None], axis=0)
    variance = np.sum((stack - mean[None]) ** 2 * weights[:, None], axis=0)
    confidence = np.sum(np.stack(coverages, axis=0) * weights[:, None], axis=0)
    return mean.astype(np.float32), variance.astype(np.float32), confidence.astype(np.float32)


@dataclass(frozen=True)
class CanonicalFieldMemory:
    """Serialized source-canonical memory consumed by target alignment."""

    manipulated_points: np.ndarray
    manipulated_dino: np.ndarray
    manipulated_field_mean: np.ndarray
    manipulated_field_var: np.ndarray
    manipulated_confidence: np.ndarray
    reference_points: np.ndarray
    reference_dino: np.ndarray
    reference_field_mean: np.ndarray
    reference_field_var: np.ndarray
    reference_confidence: np.ndarray

    def validate(self) -> None:
        for role in ("manipulated", "reference"):
            points = getattr(self, role + "_points")
            dino = getattr(self, role + "_dino")
            field = getattr(self, role + "_field_mean")
            variance = getattr(self, role + "_field_var")
            confidence = getattr(self, role + "_confidence")
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"{role} canonical points must be [N,3], got {points.shape}")
            if dino.ndim != 2 or dino.shape[0] != points.shape[0]:
                raise ValueError(f"{role} canonical DINO does not align with points")
            if any(np.asarray(value).reshape(-1).shape[0] != points.shape[0] for value in (field, variance, confidence)):
                raise ValueError(f"{role} canonical field arrays do not align with points")
            if not all(np.isfinite(value).all() for value in (points, dino, field, variance, confidence)):
                raise ValueError(f"{role} canonical memory contains NaN/Inf")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            manipulated_points=self.manipulated_points,
            manipulated_dino=self.manipulated_dino,
            manipulated_field_mean=self.manipulated_field_mean,
            manipulated_field_var=self.manipulated_field_var,
            manipulated_confidence=self.manipulated_confidence,
            reference_points=self.reference_points,
            reference_dino=self.reference_dino,
            reference_field_mean=self.reference_field_mean,
            reference_field_var=self.reference_field_var,
            reference_confidence=self.reference_confidence,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalFieldMemory":
        with np.load(Path(path), allow_pickle=False) as data:
            required = (
                "manipulated_points", "manipulated_dino", "manipulated_field_mean",
                "manipulated_field_var", "manipulated_confidence", "reference_points",
                "reference_dino", "reference_field_mean", "reference_field_var",
                "reference_confidence",
            )
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"Canonical memory missing fields: {missing}")
            memory = cls(**{key: np.asarray(data[key], dtype=np.float32) for key in required})
        memory.validate()
        return memory

