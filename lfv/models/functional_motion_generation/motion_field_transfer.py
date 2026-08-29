"""Semantic/structural transport of a remembered Stage 2 motion field.

The transport reuses the same FGW implementation used by Stage 1.  A memory
stores complete 256-point object/reference clouds, their DINO descriptors and
the corresponding relevance distributions produced by a motion-field
checkpoint.  At inference those fields are transported independently to the
current 256-point clouds and can be fused with the current encoder fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lfv.affordance_transfer.fgw_contact_transfer import (
    farthest_point_indices,
    interpolate_node_heat,
    normalized_knn_geodesic,
    solve_fgw,
)


@dataclass(frozen=True)
class MotionFieldMemory:
    manipulated_points: np.ndarray
    manipulated_dino: np.ndarray
    manipulated_field: np.ndarray
    reference_points: np.ndarray
    reference_dino: np.ndarray
    reference_field: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "MotionFieldMemory":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            aliases = {
                "manipulated_points": ("manipulated_points", "source_manipulated_points"),
                "manipulated_dino": ("manipulated_dino", "source_manipulated_dino"),
                "manipulated_field": ("manipulated_motion_field", "source_manipulated_motion_field"),
                "reference_points": ("reference_points", "source_reference_points"),
                "reference_dino": ("reference_dino", "source_reference_dino"),
                "reference_field": ("reference_motion_field", "source_reference_motion_field"),
            }
            values: dict[str, np.ndarray] = {}
            for name, candidates in aliases.items():
                key = next((candidate for candidate in candidates if candidate in data), None)
                if key is None:
                    raise KeyError(f"Motion memory {path} is missing {name}; available={data.files}")
                values[name] = np.asarray(data[key], dtype=np.float32)
        memory = cls(**values)
        for prefix in ("manipulated", "reference"):
            points = getattr(memory, f"{prefix}_points")
            dino = getattr(memory, f"{prefix}_dino")
            field = getattr(memory, f"{prefix}_field").reshape(-1)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(dino) or len(points) != len(field):
                raise ValueError(f"Invalid {prefix} memory shapes: points={points.shape}, dino={dino.shape}, field={field.shape}")
        return memory


@dataclass(frozen=True)
class MotionFieldTransferResult:
    target_field: np.ndarray
    transport: np.ndarray
    semantic_cost: np.ndarray
    source_geodesic: np.ndarray
    target_geodesic: np.ndarray
    confidence: float


def _as_distribution(field: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(field, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(values.sum())
    if total <= 1e-12:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=np.float32)
    return (values / total).astype(np.float32)


def transport_motion_field(
    source_points: np.ndarray,
    source_dino: np.ndarray,
    source_field: np.ndarray,
    target_points: np.ndarray,
    target_dino: np.ndarray,
    *,
    node_count: int = 256,
    alpha: float = 0.5,
    graph_neighbors: int = 10,
    graph_maximum_neighbors: int = 24,
    edge_length_ratio: float = 4.0,
    maximum_iterations: int = 200,
    tolerance: float = 1e-9,
    interpolation_neighbors: int = 3,
) -> MotionFieldTransferResult:
    """Transport one source relevance field to the target point cloud."""

    source_points = np.asarray(source_points, dtype=np.float32)
    target_points = np.asarray(target_points, dtype=np.float32)
    source_dino = np.asarray(source_dino, dtype=np.float32)
    target_dino = np.asarray(target_dino, dtype=np.float32)
    source_field = _as_distribution(source_field)
    if len(source_points) != len(source_dino) or len(source_points) != len(source_field):
        raise ValueError("Source points, DINO and motion field must have the same length")
    if len(target_points) != len(target_dino):
        raise ValueError("Target points and DINO must have the same length")
    source_idx = farthest_point_indices(source_points, min(node_count, len(source_points)), seed=0)
    target_idx = farthest_point_indices(target_points, min(node_count, len(target_points)), seed=1)
    source_nodes = source_points[source_idx]
    target_nodes = target_points[target_idx]
    source_structure, _ = normalized_knn_geodesic(
        source_nodes,
        neighbors=graph_neighbors,
        maximum_neighbors=graph_maximum_neighbors,
        edge_length_ratio=edge_length_ratio,
    )
    target_structure, _ = normalized_knn_geodesic(
        target_nodes,
        neighbors=graph_neighbors,
        maximum_neighbors=graph_maximum_neighbors,
        edge_length_ratio=edge_length_ratio,
    )
    result = solve_fgw(
        source_dino[source_idx],
        target_dino[target_idx],
        source_structure,
        target_structure,
        source_field[source_idx],
        alpha=alpha,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    target_nodes_field = _as_distribution(result.target_node_heat)
    target_field = interpolate_node_heat(
        target_nodes,
        target_nodes_field,
        target_points,
        neighbors=interpolation_neighbors,
    )
    target_field = _as_distribution(target_field)
    entropy = -float(np.sum(target_field * np.log(np.maximum(target_field, 1e-12)))) / np.log(max(len(target_field), 2))
    confidence = float(np.clip((1.0 - entropy) * (1.0 / (1.0 + max(result.objective, 0.0))), 0.0, 1.0))
    return MotionFieldTransferResult(
        target_field=target_field,
        transport=result.transport,
        semantic_cost=result.semantic_cost,
        source_geodesic=result.source_geodesic,
        target_geodesic=result.target_geodesic,
        confidence=confidence,
    )
