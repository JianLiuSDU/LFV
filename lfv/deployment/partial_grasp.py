"""Single-view grasp hypotheses for partially observed point clouds.

This module deliberately does not pretend that a partial point cloud contains
the occluded finger contact.  It estimates a small set of *contact-pair*
hypotheses: one endpoint is supported by the observed, heat-weighted surface
and the second endpoint is a symmetry/thickness hypothesis.  A real GraspNet
runner can be plugged in through :class:`ExternalGraspNetBackend`; this module
is the deterministic fallback used when that binary/checkpoint is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ContactPairHypothesis:
    first_contact_camera: np.ndarray
    second_contact_camera: np.ndarray
    tcp_camera: np.ndarray
    closing_axis_camera: np.ndarray
    approach_axis_camera: np.ndarray
    width_m: float
    score: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("first_contact_camera", "second_contact_camera", "tcp_camera", "closing_axis_camera", "approach_axis_camera"):
            out[key] = np.asarray(out[key]).tolist()
        return out


def _unit(x: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(x))
    if n < 1e-7:
        if fallback is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return _unit(fallback)
    return x / n


def _tcp_from_pair(center: np.ndarray, closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Construct a right-handed TCP frame (x lateral, y closing, z approach)."""
    a = _unit(approach, np.array([0.0, -1.0, 0.0], dtype=np.float32))
    c = _unit(closing, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = _unit(c - a * float(c @ a), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    x = _unit(np.cross(c, a), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y = _unit(np.cross(a, x), c)
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = np.stack((x, y, a), axis=1)
    t[:3, 3] = np.asarray(center, dtype=np.float32)
    return t


def _weighted_center(points: np.ndarray, heat: np.ndarray) -> tuple[np.ndarray, float]:
    p = np.asarray(points, dtype=np.float32)
    h = np.clip(np.asarray(heat, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if len(p) == 0:
        raise ValueError("points must not be empty")
    w = h + 1e-3
    return (p * w[:, None]).sum(0) / float(w.sum()), float(h.max())


def _candidate_axes(points: np.ndarray, approach: np.ndarray) -> list[np.ndarray]:
    p = np.asarray(points, dtype=np.float32)
    a = _unit(approach)
    centered = p - p.mean(0, keepdims=True)
    if len(p) >= 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        raw = [vt[0], vt[1], vt[2] if vt.shape[0] > 2 else vt[0]]
    else:
        raw = [np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)]
    # Include the image-horizontal direction as a stable hypothesis for a
    # front-facing camera, while removing any component along the approach.
    raw.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    raw.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    axes: list[np.ndarray] = []
    for v in raw:
        q = v - a * float(v @ a)
        if np.linalg.norm(q) < 1e-5:
            continue
        q = _unit(q)
        if not any(abs(float(q @ old)) > 0.96 for old in axes):
            axes.extend([q, -q])
    return axes


def _local_support(points: np.ndarray, endpoint: np.ndarray, radius: float) -> float:
    d = np.linalg.norm(np.asarray(points) - endpoint[None], axis=1)
    return float(np.exp(-float(np.min(d)) / max(radius, 1e-5)))


def build_contact_pair_hypotheses(
    points_camera: np.ndarray,
    heat_camera: np.ndarray,
    *,
    preferred_approach_camera: Iterable[float] = (0.0, -1.0, 0.0),
    width_candidates_m: Iterable[float] = (0.025, 0.035, 0.045, 0.055),
    top_k: int = 8,
    heat_quantile: float = 0.75,
) -> list[ContactPairHypothesis]:
    """Generate contact-pair/top-down hypotheses from a visible cloud.

    The high heat endpoint is constrained to observed points.  Its counterpart
    is allowed to be virtual, representing the occluded side of the object.
    Widths and in-plane axes are intentionally few and deterministic so the
    result is reproducible and easy to replace with a learned/GraspNet backend.
    """
    p = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    h = np.asarray(heat_camera, dtype=np.float32).reshape(-1)
    if len(p) != len(h) or len(p) == 0:
        raise ValueError("points_camera and heat_camera must have matching non-empty lengths")
    finite = np.isfinite(p).all(1) & np.isfinite(h)
    p, h = p[finite], np.clip(h[finite], 0.0, 1.0)
    if len(p) == 0:
        raise ValueError("no finite points")
    approach = _unit(np.asarray(preferred_approach_camera, dtype=np.float32))
    center, peak = _weighted_center(p, h)
    mask = h >= max(float(np.quantile(h, heat_quantile)), 0.25 * peak)
    focus = p[mask] if int(mask.sum()) >= 3 else p
    axes = _candidate_axes(focus, approach)
    # A robust object scale keeps the default widths sensible for both mugs
    # and handles; explicit width_candidates remain the primary control.
    diameter = float(np.max(np.linalg.norm(p[:, None] - p[None, :], axis=-1))) if len(p) < 600 else float(np.linalg.norm(np.ptp(p, axis=0)))
    radius = max(0.006, 0.03 * diameter)
    candidates: list[ContactPairHypothesis] = []
    for axis in axes:
        for width in width_candidates_m:
            width = float(width)
            # Symmetric pair around the heat-weighted center; choose the
            # endpoint closer to the high-heat visible surface as first contact.
            p0, p1 = center - 0.5 * width * axis, center + 0.5 * width * axis
            s0 = _local_support(p, p0, radius)
            s1 = _local_support(p, p1, radius)
            if s1 > s0:
                first, second, support = p1, p0, s1
            else:
                first, second, support = p0, p1, s0
            tcp = _tcp_from_pair((first + second) * 0.5, second - first, approach)
            # Heat should explain the observed endpoint; top-down is fixed by
            # construction but still reported as an explicit quality term.
            topdown = max(0.0, float(tcp[:3, 2] @ approach))
            score = 0.60 * peak + 0.25 * support + 0.15 * topdown
            candidates.append(ContactPairHypothesis(first, second, tcp, _unit(second - first), approach, width, score, {"backend": "contact_pair_symmetry", "first_endpoint_support": support, "heat_peak": peak, "virtual_second_contact": True}))
    candidates.sort(key=lambda x: x.score, reverse=True)
    # Remove near duplicates (same center and closing direction).
    unique: list[ContactPairHypothesis] = []
    for c in candidates:
        if any(np.linalg.norm(c.tcp_camera[:3, 3] - u.tcp_camera[:3, 3]) < 0.004 and abs(float(c.closing_axis_camera @ u.closing_axis_camera)) > 0.98 for u in unique):
            continue
        unique.append(c)
        if len(unique) >= int(top_k):
            break
    return unique


def evaluate_contact_pair_against_full_cloud(
    hypotheses: Iterable[ContactPairHypothesis],
    full_points_camera: np.ndarray,
    *,
    support_radius_m: float = 0.018,
) -> list[dict[str, Any]]:
    """Offline oracle-style evaluation using a simulator/full cloud.

    This is *only* an evaluation aid; ``full_points_camera`` is never needed
    to produce hypotheses at deployment time.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(full_points_camera, dtype=np.float32).reshape(-1, 3))
    rows: list[dict[str, Any]] = []
    for rank, h in enumerate(hypotheses):
        d0 = float(tree.query(h.first_contact_camera)[0])
        d1 = float(tree.query(h.second_contact_camera)[0])
        approach = _unit(h.approach_axis_camera)
        rows.append({"rank": rank, "score": float(h.score), "first_contact_distance_m": d0, "second_contact_distance_m": d1, "pair_supported": bool(d0 <= support_radius_m and d1 <= support_radius_m), "first_supported": bool(d0 <= support_radius_m), "second_supported": bool(d1 <= support_radius_m), "topdown_alignment": float(h.tcp_camera[:3, 2] @ approach), "width_m": float(h.width_m), "metadata": h.metadata})
    return rows

