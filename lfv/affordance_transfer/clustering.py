from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    cluster_mass: np.ndarray


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def weighted_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    weights: np.ndarray | None = None,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> KMeansResult:
    """Deterministic NumPy weighted K-Means for small patch sets."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be [N,D].")
    n_samples = features.shape[0]
    if not 1 <= n_clusters <= n_samples:
        raise ValueError(f"n_clusters={n_clusters} is invalid for N={n_samples}.")
    features = _normalize_rows(features)
    if weights is None:
        weights = np.ones(n_samples, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_samples,) or np.any(weights < 0) or not np.all(
        np.isfinite(weights)
    ):
        raise ValueError("weights must be finite, non-negative, and have shape [N].")
    if float(weights.sum()) <= 0:
        raise ValueError("weights must have positive total mass.")

    best: KMeansResult | None = None
    for init_index in range(n_init):
        rng = np.random.default_rng(seed + 104729 * init_index)
        centers = np.empty((n_clusters, features.shape[1]), dtype=np.float32)
        first = int(rng.choice(n_samples, p=weights / weights.sum()))
        centers[0] = features[first]
        closest = np.sum((features - centers[0]) ** 2, axis=1)
        chosen = {first}
        for cluster_index in range(1, n_clusters):
            probabilities = weights * closest
            probabilities[list(chosen)] = 0
            if float(probabilities.sum()) <= 1e-15:
                remaining = np.array(sorted(set(range(n_samples)) - chosen))
                next_index = int(rng.choice(remaining))
            else:
                next_index = int(rng.choice(n_samples, p=probabilities / probabilities.sum()))
            chosen.add(next_index)
            centers[cluster_index] = features[next_index]
            distance = np.sum((features - centers[cluster_index]) ** 2, axis=1)
            closest = np.minimum(closest, distance)

        previous_labels: np.ndarray | None = None
        for _ in range(max_iter):
            distances = np.sum(
                (features[:, None, :] - centers[None, :, :]) ** 2, axis=-1
            )
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for cluster_index in range(n_clusters):
                members = labels == cluster_index
                mass = float(weights[members].sum())
                if mass <= 1e-15:
                    residual = weights * distances[np.arange(n_samples), labels]
                    replacement = int(np.argmax(residual))
                    new_centers[cluster_index] = features[replacement]
                else:
                    new_centers[cluster_index] = np.average(
                        features[members], axis=0, weights=weights[members]
                    )
            new_centers = _normalize_rows(new_centers).astype(np.float32)
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            if shift <= tol:
                break
            previous_labels = labels.copy()

        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        inertia = float(np.sum(weights * distances[np.arange(n_samples), labels]))
        mass = np.bincount(labels, weights=weights, minlength=n_clusters).astype(np.float64)
        result = KMeansResult(centers, labels, inertia, mass)
        if best is None or result.inertia < best.inertia:
            best = result
    assert best is not None
    return best
