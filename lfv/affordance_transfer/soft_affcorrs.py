from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clustering import KMeansResult, weighted_kmeans


def _softmax(values: np.ndarray, axis: int, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Softmax temperature must be positive.")
    scaled = values.astype(np.float64) / temperature
    scaled -= np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(scaled)
    return (exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-15)).astype(
        np.float32
    )


@dataclass(frozen=True)
class SoftAffCorrsOutput:
    source_clustering: KMeansResult
    target_clustering: KMeansResult
    source_positive_mask: np.ndarray
    source_cluster_weights: np.ndarray
    forward_probabilities: np.ndarray
    forward_votes: np.ndarray
    backward_probabilities: np.ndarray
    backward_scores: np.ndarray
    target_cluster_scores: np.ndarray
    retained_heat_mass: float
    source_heat_distribution: np.ndarray


def soft_heatmap_affcorrs(
    source_features: np.ndarray,
    source_heat: np.ndarray,
    target_features: np.ndarray,
    *,
    source_clusters: int = 6,
    target_clusters: int = 64,
    positive_threshold: float = 0.2,
    forward_temperature: float = 0.1,
    backward_temperature: float = 0.05,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 100,
) -> SoftAffCorrsOutput:
    """Compute the continuous, bidirectionally verified AffCorrs score."""
    source_features = np.asarray(source_features, dtype=np.float32)
    target_features = np.asarray(target_features, dtype=np.float32)
    source_heat = np.asarray(source_heat, dtype=np.float32)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("Source and target features must both be [N,D].")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions differ.")
    if source_heat.shape != (source_features.shape[0],):
        raise ValueError("source_heat must have shape [N_source].")
    if np.any(source_heat < 0) or not np.all(np.isfinite(source_heat)):
        raise ValueError("source_heat must be finite and non-negative.")
    heat_sum = float(source_heat.sum())
    if heat_sum <= 1e-12:
        raise ValueError("source_heat has no positive mass.")

    source_features = source_features / np.maximum(
        np.linalg.norm(source_features, axis=1, keepdims=True), 1e-12
    )
    target_features = target_features / np.maximum(
        np.linalg.norm(target_features, axis=1, keepdims=True), 1e-12
    )
    positive = source_heat > positive_threshold
    n_positive = int(positive.sum())
    if n_positive == 0:
        raise ValueError(
            f"No source patch is above positive_threshold={positive_threshold}."
        )
    source_clusters = min(source_clusters, n_positive)
    target_clusters = min(target_clusters, target_features.shape[0])
    retained_heat_mass = float(source_heat[positive].sum() / heat_sum)

    source_fit = weighted_kmeans(
        source_features[positive],
        source_clusters,
        weights=source_heat[positive],
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    target_fit = weighted_kmeans(
        target_features,
        target_clusters,
        seed=seed + 1,
        n_init=n_init,
        max_iter=max_iter,
    )
    omega = source_fit.cluster_mass / np.maximum(source_fit.cluster_mass.sum(), 1e-15)
    omega = omega.astype(np.float32)

    forward_similarity = source_fit.centers @ target_fit.centers.T
    forward_probabilities = _softmax(
        forward_similarity, axis=1, temperature=forward_temperature
    )
    forward_votes = (omega[:, None] * forward_probabilities).sum(axis=0)

    backward_similarity = target_fit.centers @ source_features.T
    backward_probabilities = _softmax(
        backward_similarity, axis=1, temperature=backward_temperature
    )
    heat_distribution = source_heat / heat_sum
    backward_scores = backward_probabilities @ heat_distribution
    cluster_scores = forward_votes * backward_scores

    return SoftAffCorrsOutput(
        source_clustering=source_fit,
        target_clustering=target_fit,
        source_positive_mask=positive,
        source_cluster_weights=omega,
        forward_probabilities=forward_probabilities,
        forward_votes=forward_votes.astype(np.float32),
        backward_probabilities=backward_probabilities,
        backward_scores=backward_scores.astype(np.float32),
        target_cluster_scores=cluster_scores.astype(np.float32),
        retained_heat_mass=retained_heat_mass,
        source_heat_distribution=heat_distribution.astype(np.float32),
    )
