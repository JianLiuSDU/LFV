from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree

from lfv.features.base import DenseFeatureExtractor

from .pipeline import SoftHeatmapAffCorrsPipeline
from .preprocessing import prepare_image
from .schema import RGBDPart, SourceContactExample, TargetObservation, TransferResult


@dataclass(frozen=True)
class PartPointCloud:
    points_camera: np.ndarray
    pixels_uv: np.ndarray
    features: np.ndarray
    heat: np.ndarray | None = None


@dataclass(frozen=True)
class FGWResult:
    transport: np.ndarray
    target_node_heat: np.ndarray
    semantic_cost: np.ndarray
    source_geodesic: np.ndarray
    target_geodesic: np.ndarray
    objective: float
    solver: str


def _get(mapping: dict[str, Any], key: str, default: Any) -> Any:
    return mapping[key] if key in mapping else default


def _sample_feature_grid(
    grid: np.ndarray, pixels_uv: np.ndarray, transform
) -> np.ndarray:
    """Bilinearly sample patch descriptors at original-image pixel locations."""

    input_xy = transform.original_to_input(pixels_uv)
    grid_xy = input_xy / float(transform.patch_size) - 0.5
    coordinates = np.stack((grid_xy[:, 1], grid_xy[:, 0]), axis=0)
    sampled = np.stack(
        [
            map_coordinates(
                grid[..., channel], coordinates, order=1, mode="nearest"
            )
            for channel in range(grid.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)
    norms = np.linalg.norm(sampled, axis=1, keepdims=True)
    return sampled / np.maximum(norms, 1e-8)


def lift_part_to_points(
    rgbd: RGBDPart,
    feature_grid: np.ndarray,
    transform,
    *,
    heatmap: np.ndarray | None = None,
    minimum_depth_m: float = 1e-4,
    maximum_depth_m: float = 2.5,
) -> PartPointCloud:
    valid = (
        rgbd.part_mask
        & np.isfinite(rgbd.depth_m)
        & (rgbd.depth_m >= minimum_depth_m)
        & (rgbd.depth_m <= maximum_depth_m)
    )
    rows, columns = np.nonzero(valid)
    if not len(columns):
        raise ValueError("Functional-part mask contains no valid aligned depth pixel.")
    z = rgbd.depth_m[rows, columns]
    intrinsic = rgbd.intrinsic_cv
    points = np.stack(
        (
            (columns.astype(np.float32) - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows.astype(np.float32) - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ),
        axis=-1,
    ).astype(np.float32)
    pixels_uv = np.stack((columns, rows), axis=-1).astype(np.int32)
    features = _sample_feature_grid(feature_grid, pixels_uv, transform)
    heat = None
    if heatmap is not None:
        heat = np.clip(
            np.asarray(heatmap, dtype=np.float32)[rows, columns], 0.0, 1.0
        )
    return PartPointCloud(points, pixels_uv, features, heat)


def farthest_point_indices(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Deterministic Euclidean FPS with a seeded first point."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {points.shape}.")
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(len(points)))
    distances = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(distances))
        candidate_distances = np.sum(
            (points - points[selected[index]]) ** 2, axis=1
        )
        distances = np.minimum(distances, candidate_distances)
    return selected


def _sample_cloud(cloud: PartPointCloud, count: int, seed: int) -> PartPointCloud:
    indices = farthest_point_indices(cloud.points_camera, count, seed)
    return PartPointCloud(
        points_camera=cloud.points_camera[indices],
        pixels_uv=cloud.pixels_uv[indices],
        features=cloud.features[indices],
        heat=None if cloud.heat is None else cloud.heat[indices],
    )


def normalized_knn_geodesic(
    points: np.ndarray,
    *,
    neighbors: int = 10,
    maximum_neighbors: int = 20,
    edge_length_ratio: float = 4.0,
    normalization_quantile: float = 0.95,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Build a connected local graph and return scale-normalized geodesics."""

    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count < 3:
        raise ValueError("At least three points are required for a geodesic graph.")
    maximum_neighbors = min(maximum_neighbors, count - 1)
    neighbors = min(neighbors, maximum_neighbors)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=maximum_neighbors + 1)
    nearest_median = float(np.median(distances[:, 1]))
    if nearest_median <= 0:
        raise ValueError("Point cloud has degenerate duplicate spacing.")
    threshold = nearest_median * edge_length_ratio
    graph = None
    components = count
    used_neighbors = neighbors
    for used_neighbors in range(neighbors, maximum_neighbors + 1, 2):
        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        for row in range(count):
            for distance, column in zip(
                distances[row, 1 : used_neighbors + 1],
                indices[row, 1 : used_neighbors + 1],
                strict=True,
            ):
                if distance <= threshold:
                    rows.extend((row, int(column)))
                    columns.extend((int(column), row))
                    weights.extend((float(distance), float(distance)))
        graph = csr_matrix((weights, (rows, columns)), shape=(count, count))
        components, _ = connected_components(graph, directed=False)
        if components == 1:
            break
    if graph is None or components != 1:
        raise ValueError(
            "Functional-part kNN graph is disconnected; enlarge the part mask, "
            "raise maximum_neighbors, or raise edge_length_ratio."
        )
    geodesic = shortest_path(graph, directed=False)
    finite_nonzero = geodesic[np.isfinite(geodesic) & (geodesic > 0)]
    scale = float(np.quantile(finite_nonzero, normalization_quantile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not determine a valid geodesic normalization scale.")
    normalized = np.clip(geodesic / scale, 0.0, 1.0).astype(np.float64)
    return normalized, {
        "neighbors": int(used_neighbors),
        "nearest_neighbor_median_m": nearest_median,
        "edge_threshold_m": threshold,
        "normalization_scale_m": scale,
    }


def _fgw_objective(
    transport: np.ndarray,
    semantic: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    alpha: float,
) -> float:
    source_mass = transport.sum(axis=1)
    target_mass = transport.sum(axis=0)
    tensor = (
        (source_structure**2) @ source_mass[:, None]
        + ((target_structure**2) @ target_mass[:, None]).T
        - 2.0 * source_structure @ transport @ target_structure.T
    )
    return float(
        (1.0 - alpha) * np.sum(semantic * transport)
        + alpha * np.sum(tensor * transport)
    )


def _fallback_equal_mass_fgw(
    semantic: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    *,
    alpha: float,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """Deterministic Frank-Wolfe fallback for equal-size uniform marginals."""

    from scipy.optimize import linear_sum_assignment

    source_count, target_count = semantic.shape
    if source_count != target_count:
        raise ImportError(
            "POT is required when source and target FGW node counts differ."
        )
    count = source_count
    transport = np.full((count, count), 1.0 / (count * count), dtype=np.float64)
    previous = _fgw_objective(
        transport, semantic, source_structure, target_structure, alpha
    )
    for _ in range(maximum_iterations):
        source_mass = transport.sum(axis=1)
        target_mass = transport.sum(axis=0)
        tensor = (
            (source_structure**2) @ source_mass[:, None]
            + ((target_structure**2) @ target_mass[:, None]).T
            - 2.0 * source_structure @ transport @ target_structure.T
        )
        gradient = (1.0 - alpha) * semantic + 2.0 * alpha * tensor
        rows, columns = linear_sum_assignment(gradient)
        vertex = np.zeros_like(transport)
        vertex[rows, columns] = 1.0 / count
        direction = vertex - transport
        f0 = previous
        f1 = _fgw_objective(
            vertex, semantic, source_structure, target_structure, alpha
        )
        midpoint = transport + 0.5 * direction
        fhalf = _fgw_objective(
            midpoint, semantic, source_structure, target_structure, alpha
        )
        quadratic = 2.0 * (f1 + f0 - 2.0 * fhalf)
        linear = f1 - f0 - quadratic
        candidates = [0.0, 1.0]
        if quadratic > 0:
            candidates.append(float(np.clip(-linear / (2.0 * quadratic), 0.0, 1.0)))
        objectives = [
            _fgw_objective(
                transport + gamma * direction,
                semantic,
                source_structure,
                target_structure,
                alpha,
            )
            for gamma in candidates
        ]
        best = int(np.argmin(objectives))
        transport = transport + candidates[best] * direction
        current = objectives[best]
        if abs(previous - current) <= tolerance * max(1.0, abs(previous)):
            previous = current
            break
        previous = current
    return transport, previous


def solve_fgw(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_structure: np.ndarray,
    target_structure: np.ndarray,
    source_heat: np.ndarray,
    *,
    alpha: float = 0.5,
    maximum_iterations: int = 200,
    tolerance: float = 1e-9,
) -> FGWResult:
    """Solve balanced FGW and directly transport the source contact field."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1].")
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    source_features /= np.maximum(
        np.linalg.norm(source_features, axis=1, keepdims=True), 1e-12
    )
    target_features /= np.maximum(
        np.linalg.norm(target_features, axis=1, keepdims=True), 1e-12
    )
    semantic = np.clip(
        0.5 * (1.0 - source_features @ target_features.T), 0.0, 1.0
    )
    source_count, target_count = semantic.shape
    source_mass = np.full(source_count, 1.0 / source_count, dtype=np.float64)
    target_mass = np.full(target_count, 1.0 / target_count, dtype=np.float64)
    try:
        import ot

        transport, log = ot.gromov.fused_gromov_wasserstein(
            semantic,
            source_structure,
            target_structure,
            source_mass,
            target_mass,
            loss_fun="square_loss",
            alpha=alpha,
            armijo=False,
            log=True,
            max_iter=maximum_iterations,
            tol_rel=tolerance,
            tol_abs=tolerance,
        )
        objective = float(log.get("fgw_dist", np.nan))
        solver = f"POT-{getattr(ot, '__version__', 'unknown')}"
    except ImportError:
        transport, objective = _fallback_equal_mass_fgw(
            semantic,
            source_structure,
            target_structure,
            alpha=alpha,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        solver = "scipy-frank-wolfe-fallback"
    column_mass = transport.sum(axis=0)
    target_heat = transport.T @ np.asarray(source_heat, dtype=np.float64)
    target_heat /= np.maximum(column_mass, 1e-12)
    return FGWResult(
        transport=np.asarray(transport, dtype=np.float32),
        target_node_heat=np.clip(target_heat, 0.0, 1.0).astype(np.float32),
        semantic_cost=semantic.astype(np.float32),
        source_geodesic=np.asarray(source_structure, dtype=np.float32),
        target_geodesic=np.asarray(target_structure, dtype=np.float32),
        objective=objective,
        solver=solver,
    )


def interpolate_node_heat(
    node_points: np.ndarray,
    node_heat: np.ndarray,
    query_points: np.ndarray,
    *,
    neighbors: int = 3,
) -> np.ndarray:
    neighbors = min(int(neighbors), len(node_points))
    distances, indices = cKDTree(node_points).query(query_points, k=neighbors)
    if neighbors == 1:
        return np.asarray(node_heat, dtype=np.float32)[indices]
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(np.asarray(node_heat)[indices] * weights, axis=1).astype(np.float32)


class AffCorrsFGWContactTransferPipeline:
    """AffCorrs semantic localization followed by 3D FGW field transport."""

    def __init__(self, extractor: DenseFeatureExtractor, config: dict[str, Any]) -> None:
        self.extractor = extractor
        self.config = config
        self.affcorrs = SoftHeatmapAffCorrsPipeline(extractor, config)

    def transfer(
        self,
        source: SourceContactExample,
        target: TargetObservation,
        source_rgbd: RGBDPart,
        target_rgbd: RGBDPart,
    ) -> TransferResult:
        baseline = self.affcorrs.transfer(source, target)
        prep_cfg = self.config.get("preprocessing", {})
        fgw_cfg = self.config.get("fgw", {})
        common = {
            "input_size": int(_get(prep_cfg, "input_size", 518)),
            "patch_size": self.extractor.patch_size,
            "bbox_margin": float(_get(prep_cfg, "bbox_margin", 0.15)),
        }
        source_prepared = prepare_image(
            source.rgb, source.mask, heatmap=source.heatmap, **common
        )
        target_prepared = prepare_image(target.rgb, target.mask, **common)
        source_features = self.extractor.extract(source_prepared.rgb)
        target_features = self.extractor.extract(target_prepared.rgb)
        source_cloud = lift_part_to_points(
            source_rgbd,
            source_features,
            source_prepared.transform,
            heatmap=source.heatmap,
            minimum_depth_m=float(_get(fgw_cfg, "minimum_depth_m", 1e-4)),
            maximum_depth_m=float(_get(fgw_cfg, "maximum_depth_m", 2.5)),
        )
        target_cloud = lift_part_to_points(
            target_rgbd,
            target_features,
            target_prepared.transform,
            minimum_depth_m=float(_get(fgw_cfg, "minimum_depth_m", 1e-4)),
            maximum_depth_m=float(_get(fgw_cfg, "maximum_depth_m", 2.5)),
        )
        node_count = int(_get(fgw_cfg, "node_count", 256))
        seed = int(_get(fgw_cfg, "seed", self.config.get("seed", 0)))
        source_nodes = _sample_cloud(source_cloud, node_count, seed)
        target_nodes = _sample_cloud(target_cloud, node_count, seed + 1)
        graph_kwargs = {
            "neighbors": int(_get(fgw_cfg, "graph_neighbors", 10)),
            "maximum_neighbors": int(_get(fgw_cfg, "graph_maximum_neighbors", 20)),
            "edge_length_ratio": float(_get(fgw_cfg, "edge_length_ratio", 4.0)),
            "normalization_quantile": float(
                _get(fgw_cfg, "geodesic_normalization_quantile", 0.95)
            ),
        }
        source_geodesic, source_graph = normalized_knn_geodesic(
            source_nodes.points_camera, **graph_kwargs
        )
        target_geodesic, target_graph = normalized_knn_geodesic(
            target_nodes.points_camera, **graph_kwargs
        )
        assert source_nodes.heat is not None
        fgw = solve_fgw(
            source_nodes.features,
            target_nodes.features,
            source_geodesic,
            target_geodesic,
            source_nodes.heat,
            alpha=float(_get(fgw_cfg, "alpha", 0.5)),
            maximum_iterations=int(_get(fgw_cfg, "maximum_iterations", 200)),
            tolerance=float(_get(fgw_cfg, "tolerance", 1e-9)),
        )
        full_target_heat = interpolate_node_heat(
            target_nodes.points_camera,
            fgw.target_node_heat,
            target_cloud.points_camera,
            neighbors=int(_get(fgw_cfg, "interpolation_neighbors", 3)),
        )
        raw = np.zeros_like(target_rgbd.depth_m, dtype=np.float32)
        uv = target_cloud.pixels_uv
        raw[uv[:, 1], uv[:, 0]] = full_target_heat

        gate_floor = float(_get(fgw_cfg, "affcorrs_gate_floor", 1.0))
        gate_power = float(_get(fgw_cfg, "affcorrs_gate_power", 0.5))
        if not 0.0 <= gate_floor <= 1.0:
            raise ValueError("affcorrs_gate_floor must lie in [0,1].")
        semantic_gate = np.power(
            np.clip(baseline.target_heatmap, 0.0, 1.0), gate_power
        )
        gated = raw * (gate_floor + (1.0 - gate_floor) * semantic_gate)
        gated *= target_rgbd.part_mask
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update(
            {
                "method": "affcorrs_fgw",
                "affcorrs_target_heatmap": baseline.target_heatmap,
                "target_heatmap_fgw_raw": raw,
                "semantic_gate": semantic_gate,
                "transport": fgw.transport,
                "semantic_cost": fgw.semantic_cost,
                "source_geodesic": fgw.source_geodesic,
                "target_geodesic": fgw.target_geodesic,
                "source_node_points_camera": source_nodes.points_camera,
                "target_node_points_camera": target_nodes.points_camera,
                "source_node_pixels_uv": source_nodes.pixels_uv,
                "target_node_pixels_uv": target_nodes.pixels_uv,
                "source_node_heat": source_nodes.heat,
                "target_node_heat": fgw.target_node_heat,
                "source_valid_part_points": int(len(source_cloud.points_camera)),
                "target_valid_part_points": int(len(target_cloud.points_camera)),
                "source_fgw_nodes": int(len(source_nodes.points_camera)),
                "target_fgw_nodes": int(len(target_nodes.points_camera)),
                "source_graph": source_graph,
                "target_graph": target_graph,
                "fgw_alpha": float(_get(fgw_cfg, "alpha", 0.5)),
                "fgw_objective": fgw.objective,
                "fgw_solver": fgw.solver,
                "affcorrs_gate_floor": gate_floor,
                "affcorrs_gate_power": gate_power,
            }
        )
        rejection_reasons = list(baseline.rejection_reasons)
        if float(gated.max(initial=0.0)) <= 1e-8:
            rejection_reasons.append("empty_fgw_target_heatmap")
        return TransferResult(
            target_heatmap=np.clip(gated, 0.0, 1.0),
            target_heatmap_raw=raw,
            confidence=baseline.confidence,
            accepted=not rejection_reasons,
            rejection_reasons=rejection_reasons,
            diagnostics=diagnostics,
        )
