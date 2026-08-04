from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .soft_affcorrs import SoftAffCorrsOutput


@dataclass(frozen=True)
class ConfidenceResult:
    values: dict[str, float]
    accepted: bool
    rejection_reasons: list[str]


def compute_transfer_confidence(
    matching: SoftAffCorrsOutput,
    target_patch_scores: np.ndarray,
    *,
    minimum_retained_heat: float = 0.5,
    minimum_cycle_score: float = 0.05,
    minimum_peak_score: float = 0.05,
    maximum_entropy: float = 0.98,
    minimum_global_score: float = 0.05,
) -> ConfidenceResult:
    """Summarize cycle consistency, saliency, and concentration.

    Backward probabilities are distributions over all source foreground
    patches.  Their heat-weighted scores are calibrated between the uniform
    baseline and the highest possible source heat mass before aggregation.
    """
    scores = np.asarray(target_patch_scores, dtype=np.float64)
    scores = np.maximum(scores, 0)
    n_source = matching.source_heat_distribution.size
    q_uniform = 1.0 / max(n_source, 1)
    q_upper = float(matching.source_heat_distribution.max())
    denominator = max(q_upper - q_uniform, 1e-12)
    calibrated_backward = np.clip(
        (matching.backward_scores - q_uniform) / denominator, 0.0, 1.0
    )
    vote_sum = float(matching.forward_votes.sum())
    cycle = float(
        np.sum(matching.forward_votes * calibrated_backward) / max(vote_sum, 1e-12)
    )

    positive_scores = scores[scores > 0]
    if positive_scores.size:
        q50, q95 = np.quantile(positive_scores, [0.5, 0.95])
        peak = float(np.clip((q95 - q50) / max(q95, 1e-12), 0.0, 1.0))
    else:
        peak = 0.0

    if float(scores.sum()) > 1e-15 and scores.size > 1:
        distribution = scores / scores.sum()
        entropy = float(
            -np.sum(distribution * np.log(np.maximum(distribution, 1e-15)))
            / math.log(scores.size)
        )
    else:
        entropy = 1.0
    concentration = float(np.clip(1.0 - entropy, 0.0, 1.0))
    retained = float(matching.retained_heat_mass)
    # The method-level global confidence is defined only by cycle consistency,
    # peak saliency, and heat concentration.  Retained source heat is reported
    # and thresholded separately so it cannot inflate the requested score.
    components = np.maximum([cycle, peak, concentration], 1e-8)
    global_score = float(np.exp(np.mean(np.log(components))))

    values = {
        "global": global_score,
        "cycle": cycle,
        "peak": peak,
        "entropy": entropy,
        "concentration": concentration,
        "retained_heat_mass": retained,
        "backward_uniform_baseline": q_uniform,
        "backward_upper_bound": q_upper,
    }
    reasons: list[str] = []
    if retained < minimum_retained_heat:
        reasons.append("insufficient_source_heat_above_threshold")
    if cycle < minimum_cycle_score:
        reasons.append("low_cycle_consistency")
    if peak < minimum_peak_score:
        reasons.append("low_target_peak_contrast")
    if entropy > maximum_entropy:
        reasons.append("target_heat_is_too_diffuse")
    if global_score < minimum_global_score:
        reasons.append("low_global_confidence")
    return ConfidenceResult(values=values, accepted=not reasons, rejection_reasons=reasons)
