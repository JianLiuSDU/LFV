from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContactHeatPropagationConfig:
    """Geometry constraints for visible-to-opposite contact propagation.

    Distances are expressed in metres.  Surface normals must be consistently
    oriented outwards when signed antipodal checks are enabled.
    """

    projection_radius: float = 0.006
    seed_quantile: float = 0.80
    max_seed_points: int = 256
    min_contact_width: float = 0.004
    max_contact_width: float = 0.045
    min_antipodal_cos: float = 0.65
    min_normal_opposition_cos: float = 0.65
    local_spread_radius: float = 0.006
    local_spread_sigma: float = 0.003
    local_normal_cos: float = 0.55
    opposite_pairs_per_seed: int = 2
    min_pair_score: float = 0.05
    hidden_distance: float = 0.006
    require_hidden_opposite: bool = False


@dataclass(frozen=True)
class AntipodalContactPair:
    visible_index: int
    opposite_index: int
    score: float
    width_m: float
    visible_alignment: float
    opposite_alignment: float
    normal_opposition: float
    opposite_visible_distance_m: float
    opposite_is_hidden: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass
class ContactHeatPropagationResult:
    visible_heat: np.ndarray
    pair_visible_heat: np.ndarray
    opposite_heat: np.ndarray
    full_heat: np.ndarray
    projected_visible_indices: np.ndarray
    projected_visible_distances: np.ndarray
    pairs: list[AntipodalContactPair]

    def summary(self) -> dict[str, float | int]:
        hidden_pairs = sum(pair.opposite_is_hidden for pair in self.pairs)
        return {
            "num_full_points": int(len(self.full_heat)),
            "num_projected_visible_points": int(len(self.projected_visible_indices)),
            "num_antipodal_pairs": int(len(self.pairs)),
            "num_hidden_antipodal_pairs": int(hidden_pairs),
            "visible_heat_max": float(self.visible_heat.max(initial=0.0)),
            "pair_visible_heat_max": float(self.pair_visible_heat.max(initial=0.0)),
            "opposite_heat_max": float(self.opposite_heat.max(initial=0.0)),
            "full_heat_max": float(self.full_heat.max(initial=0.0)),
            "visible_hot_points": int(np.count_nonzero(self.visible_heat > 0.1)),
            "opposite_hot_points": int(np.count_nonzero(self.opposite_heat > 0.1)),
        }


def _as_points(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(values):
        return values
    low = float(np.quantile(values, 0.02))
    high = float(np.quantile(values, 0.98))
    if high - low < 1e-8:
        return np.clip(values, 0.0, 1.0)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _spread_on_same_surface(
    heat: np.ndarray,
    source_index: int,
    source_value: float,
    points: np.ndarray,
    normals: np.ndarray,
    tree: cKDTree,
    config: ContactHeatPropagationConfig,
) -> None:
    neighbor_indices = np.asarray(
        tree.query_ball_point(points[source_index], config.local_spread_radius),
        dtype=np.int64,
    )
    if not len(neighbor_indices):
        return
    normal_alignment = normals[neighbor_indices] @ normals[source_index]
    keep = normal_alignment >= config.local_normal_cos
    neighbor_indices = neighbor_indices[keep]
    if not len(neighbor_indices):
        return
    distances = np.linalg.norm(
        points[neighbor_indices] - points[source_index][None],
        axis=-1,
    )
    weights = np.exp(
        -0.5 * np.square(distances / max(config.local_spread_sigma, 1e-8))
    )
    heat[neighbor_indices] = np.maximum(
        heat[neighbor_indices],
        float(source_value) * weights.astype(np.float32),
    )


def propagate_contact_heat_to_opposite_surface(
    visible_points: np.ndarray,
    visible_heat: np.ndarray,
    full_points: np.ndarray,
    full_normals: np.ndarray,
    *,
    config: ContactHeatPropagationConfig | None = None,
) -> ContactHeatPropagationResult:
    """Propagate visible affordance heat through valid antipodal contact pairs.

    This function deliberately does not diffuse heat directly by Euclidean
    proximity across object thickness.  It first projects visible heat to the
    complete surface, then finds an opposite point whose outward normal and
    contact chord satisfy signed parallel-jaw antipodal constraints.
    """

    config = config or ContactHeatPropagationConfig()
    visible_points = _as_points("visible_points", visible_points)
    full_points = _as_points("full_points", full_points)
    full_normals = _normalize_vectors(_as_points("full_normals", full_normals))
    if len(full_points) != len(full_normals):
        raise ValueError("full_points and full_normals must contain the same number of points")
    visible_heat = np.asarray(visible_heat, dtype=np.float32).reshape(-1)
    if len(visible_heat) != len(visible_points):
        raise ValueError("visible_heat and visible_points must contain the same number of points")
    visible_heat = _robust_unit_interval(visible_heat)
    if not len(full_points):
        raise ValueError("full_points cannot be empty")

    full_tree = cKDTree(full_points)
    projection_distance, projection_index = full_tree.query(visible_points, k=1)
    projection_distance = np.asarray(projection_distance, dtype=np.float32)
    projection_index = np.asarray(projection_index, dtype=np.int64)
    projection_ok = projection_distance <= config.projection_radius
    accepted_indices = projection_index[projection_ok]
    accepted_distances = projection_distance[projection_ok]

    projected_heat = np.zeros(len(full_points), dtype=np.float32)
    for full_index, heat_value in zip(
        accepted_indices,
        visible_heat[projection_ok],
        strict=True,
    ):
        projected_heat[full_index] = max(projected_heat[full_index], float(heat_value))

    visible_full_heat = np.zeros_like(projected_heat)
    nonzero_projected = np.flatnonzero(projected_heat > 0)
    for full_index in nonzero_projected:
        _spread_on_same_surface(
            visible_full_heat,
            int(full_index),
            float(projected_heat[full_index]),
            full_points,
            full_normals,
            full_tree,
            config,
        )

    if not np.any(visible_full_heat > 0):
        return ContactHeatPropagationResult(
            visible_heat=visible_full_heat,
            pair_visible_heat=np.zeros_like(visible_full_heat),
            opposite_heat=np.zeros_like(visible_full_heat),
            full_heat=visible_full_heat.copy(),
            projected_visible_indices=accepted_indices,
            projected_visible_distances=accepted_distances,
            pairs=[],
        )

    nonzero_values = visible_full_heat[visible_full_heat > 0]
    seed_threshold = float(np.quantile(nonzero_values, config.seed_quantile))
    seed_indices = np.flatnonzero(visible_full_heat >= seed_threshold)
    if len(seed_indices) > config.max_seed_points:
        order = np.argsort(visible_full_heat[seed_indices])[::-1]
        seed_indices = seed_indices[order[: config.max_seed_points]]

    visible_tree = cKDTree(visible_points)
    opposite_heat = np.zeros_like(visible_full_heat)
    pair_visible_heat = np.zeros_like(visible_full_heat)
    pairs: list[AntipodalContactPair] = []
    for visible_index in seed_indices:
        p = full_points[visible_index]
        normal_p = full_normals[visible_index]
        candidates = np.asarray(
            full_tree.query_ball_point(p, config.max_contact_width),
            dtype=np.int64,
        )
        if not len(candidates):
            continue
        offsets = full_points[candidates] - p[None]
        widths = np.linalg.norm(offsets, axis=-1)
        keep = widths >= config.min_contact_width
        candidates = candidates[keep]
        offsets = offsets[keep]
        widths = widths[keep]
        if not len(candidates):
            continue

        chords = offsets / np.maximum(widths[:, None], 1e-8)
        # For c pointing p -> q, the outward normals of an ideal pair are
        # n_p ~= -c and n_q ~= +c.
        visible_alignment = -(chords @ normal_p)
        opposite_alignment = np.einsum(
            "ij,ij->i",
            full_normals[candidates],
            chords,
        )
        normal_opposition = -(full_normals[candidates] @ normal_p)
        valid = (
            (visible_alignment >= config.min_antipodal_cos)
            & (opposite_alignment >= config.min_antipodal_cos)
            & (normal_opposition >= config.min_normal_opposition_cos)
        )
        if not np.any(valid):
            continue
        candidates = candidates[valid]
        widths = widths[valid]
        visible_alignment = visible_alignment[valid]
        opposite_alignment = opposite_alignment[valid]
        normal_opposition = normal_opposition[valid]
        hidden_distance, _ = visible_tree.query(full_points[candidates], k=1)
        hidden_distance = np.asarray(hidden_distance, dtype=np.float32)
        hidden = hidden_distance >= config.hidden_distance
        if config.require_hidden_opposite:
            keep_hidden = hidden
            candidates = candidates[keep_hidden]
            widths = widths[keep_hidden]
            visible_alignment = visible_alignment[keep_hidden]
            opposite_alignment = opposite_alignment[keep_hidden]
            normal_opposition = normal_opposition[keep_hidden]
            hidden_distance = hidden_distance[keep_hidden]
            hidden = hidden[keep_hidden]
            if not len(candidates):
                continue

        width_preference = np.exp(
            -0.15 * widths / max(config.max_contact_width, 1e-8)
        )
        pair_scores = (
            visible_full_heat[visible_index]
            * visible_alignment
            * opposite_alignment
            * normal_opposition
            * width_preference
        )
        order = np.argsort(pair_scores)[::-1]
        kept = 0
        for local_index in order:
            score = float(pair_scores[local_index])
            if score < config.min_pair_score:
                continue
            opposite_index = int(candidates[local_index])
            pair = AntipodalContactPair(
                visible_index=int(visible_index),
                opposite_index=opposite_index,
                score=score,
                width_m=float(widths[local_index]),
                visible_alignment=float(visible_alignment[local_index]),
                opposite_alignment=float(opposite_alignment[local_index]),
                normal_opposition=float(normal_opposition[local_index]),
                opposite_visible_distance_m=float(hidden_distance[local_index]),
                opposite_is_hidden=bool(hidden[local_index]),
            )
            pairs.append(pair)
            _spread_on_same_surface(
                pair_visible_heat,
                int(visible_index),
                score,
                full_points,
                full_normals,
                full_tree,
                config,
            )
            _spread_on_same_surface(
                opposite_heat,
                opposite_index,
                score,
                full_points,
                full_normals,
                full_tree,
                config,
            )
            kept += 1
            if kept >= config.opposite_pairs_per_seed:
                break

    # The physical contact field contains only surface regions that participate
    # in at least one valid antipodal pair.  Unpaired semantic heat remains
    # available separately in ``visible_heat`` for diagnosis.
    full_heat = np.maximum(pair_visible_heat, opposite_heat)
    return ContactHeatPropagationResult(
        visible_heat=visible_full_heat,
        pair_visible_heat=pair_visible_heat,
        opposite_heat=opposite_heat,
        full_heat=full_heat,
        projected_visible_indices=accepted_indices,
        projected_visible_distances=accepted_distances,
        pairs=pairs,
    )
