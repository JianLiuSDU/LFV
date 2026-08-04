#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.grasping import CollisionLimits, strict_collision_mask


def _install_graspnet_imports(graspnet_root: Path):
    for path in [
        graspnet_root,
        graspnet_root / "models",
        graspnet_root / "dataset",
        graspnet_root / "utils",
    ]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup

    return GraspNet, pred_decode, GraspGroup, ModelFreeCollisionDetector


def _grasp_points(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    width = float(max(row[1], 0.015))
    depth = float(max(row[3], 0.025))
    rotation = row[4:13].reshape(3, 3)
    center = row[13:16]
    local = np.asarray(
        [
            [-0.020, -width / 2, 0],
            [depth, -width / 2, 0],
            [-0.020, width / 2, 0],
            [depth, width / 2, 0],
            [0, 0, 0],
            [depth, 0, 0],
        ],
        dtype=np.float32,
    )
    return (rotation @ local.T).T + center[None]


def _transform_grasp_row(
    row: np.ndarray,
    source_to_target: np.ndarray,
) -> np.ndarray:
    transformed = np.asarray(row, dtype=np.float32).copy()
    source_to_target = np.asarray(source_to_target, dtype=np.float32)
    transformed[4:13] = (
        source_to_target[:3, :3] @ row[4:13].reshape(3, 3)
    ).reshape(-1)
    transformed[13:16] = (
        source_to_target[:3, :3] @ row[13:16]
        + source_to_target[:3, 3]
    )
    return transformed


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        (
            intrinsic[0, 0] * points[:, 0] / z + intrinsic[0, 2],
            intrinsic[1, 1] * points[:, 1] / z + intrinsic[1, 2],
        ),
        axis=-1,
    )


def _draw_grasp(
    canvas: np.ndarray,
    row: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    label: str,
) -> None:
    points = _grasp_points(row)
    if np.any(points[:, 2] <= 0):
        return
    pixels = _project(points, intrinsic).round().astype(np.int32)
    left_base, left_tip, right_base, right_tip, center, approach = [
        tuple(pixel) for pixel in pixels
    ]
    cv2.line(canvas, left_base, left_tip, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, right_base, right_tip, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, left_base, right_base, color, thickness, cv2.LINE_AA)
    cv2.arrowedLine(
        canvas,
        center,
        approach,
        (255, 150, 40),
        thickness,
        cv2.LINE_AA,
        tipLength=0.25,
    )
    cv2.putText(
        canvas,
        label,
        (center[0] + 5, center[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _compose_graspnet_input(
    full_points: np.ndarray,
    full_heat: np.ndarray,
    full_color: np.ndarray,
    scene_background_points: np.ndarray,
    scene_background_colors: np.ndarray,
    *,
    num_points: int,
    target_ratio: float,
    heat_threshold: float,
    rng: np.random.Generator,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, dict]:
    hot = full_heat >= heat_threshold
    if not np.any(hot):
        raise RuntimeError(
            f"No complete-geometry point exceeds heat threshold {heat_threshold:.3f}"
        )
    points = np.concatenate((full_points, scene_background_points), axis=0)
    colors = np.concatenate(
        (
            np.broadcast_to(full_color[None], (len(full_points), 3)),
            scene_background_colors,
        ),
        axis=0,
    ).astype(np.float32)
    heat = np.concatenate(
        (full_heat, np.zeros(len(scene_background_points), dtype=np.float32)),
        axis=0,
    )
    target_indices = np.flatnonzero(heat >= heat_threshold)
    context_indices = np.flatnonzero(heat < heat_threshold)
    target_count = min(num_points - 1, int(round(num_points * target_ratio)))
    target_weights = np.maximum(heat[target_indices], 1e-4).astype(np.float64)
    target_weights /= target_weights.sum()
    sampled_target = rng.choice(
        target_indices,
        target_count,
        replace=len(target_indices) < target_count,
        p=target_weights,
    )

    target_center = np.average(
        full_points[hot],
        axis=0,
        weights=np.maximum(full_heat[hot], 1e-4),
    )
    context_count = num_points - target_count
    context_distance = np.linalg.norm(points[context_indices] - target_center[None], axis=-1)
    pool_count = min(len(context_indices), max(context_count, context_count * 4))
    pool = context_indices[np.argpartition(context_distance, pool_count - 1)[:pool_count]]
    sampled_context = rng.choice(
        pool,
        context_count,
        replace=len(pool) < context_count,
    )
    selected = np.concatenate((sampled_target, sampled_context))
    rng.shuffle(selected)
    sampled_points = points[selected].astype(np.float32)
    sampled_colors = colors[selected].astype(np.float32)
    sampled_workspace = (heat[selected] >= heat_threshold).astype(np.float32)
    endpoints = {
        "point_clouds": torch.from_numpy(sampled_points).unsqueeze(0).cuda(),
        "cloud_colors": torch.from_numpy(sampled_colors).unsqueeze(0).cuda(),
        "workspace_mask": torch.from_numpy(sampled_workspace).unsqueeze(0).cuda(),
    }
    debug = {
        "num_input_points": int(num_points),
        "num_hot_full_points": int(hot.sum()),
        "num_sampled_target_points": int(target_count),
        "num_sampled_context_points": int(context_count),
        "target_center_camera": target_center.astype(float).tolist(),
        "heat_threshold": float(heat_threshold),
        "target_ratio": float(target_ratio),
    }
    return endpoints, points, colors, debug


def _rank_by_contact(
    candidates: np.ndarray,
    full_points: np.ndarray,
    full_heat: np.ndarray,
    pair_array: np.ndarray,
    *,
    desired_approach: np.ndarray | None = None,
    desired_closing: np.ndarray | None = None,
    preferred_contact_center: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    if desired_approach is not None:
        desired_approach = np.asarray(desired_approach, dtype=np.float32)
        desired_approach /= max(float(np.linalg.norm(desired_approach)), 1e-8)
    if desired_closing is not None:
        desired_closing = np.asarray(desired_closing, dtype=np.float32)
        desired_closing /= max(float(np.linalg.norm(desired_closing)), 1e-8)
    if preferred_contact_center is not None:
        preferred_contact_center = np.asarray(
            preferred_contact_center, dtype=np.float32
        ).reshape(3)
    point_tree = cKDTree(full_points)
    if len(pair_array):
        visible_indices = pair_array[:, 0].astype(np.int64)
        opposite_indices = pair_array[:, 1].astype(np.int64)
        pair_centers = 0.5 * (
            full_points[visible_indices] + full_points[opposite_indices]
        )
        pair_chords = full_points[opposite_indices] - full_points[visible_indices]
        pair_chords /= np.maximum(np.linalg.norm(pair_chords, axis=-1, keepdims=True), 1e-8)
        pair_scores = pair_array[:, 2]
        pair_tree = cKDTree(pair_centers)
    else:
        pair_centers = np.zeros((0, 3), dtype=np.float32)
        pair_chords = np.zeros((0, 3), dtype=np.float32)
        pair_scores = np.zeros(0, dtype=np.float32)
        pair_tree = None

    raw_scores = candidates[:, 0]
    score_min = float(raw_scores.min(initial=0.0))
    score_max = float(raw_scores.max(initial=1.0))
    score_norm = (raw_scores - score_min) / max(score_max - score_min, 1e-8)
    records = []
    for index, row in enumerate(candidates):
        gripper_points = _grasp_points(row)
        tips = gripper_points[[1, 3]]
        contact_center = tips.mean(axis=0)
        preferred_center_distance = (
            None
            if preferred_contact_center is None
            else float(np.linalg.norm(contact_center - preferred_contact_center))
        )
        preferred_center_score = (
            None
            if preferred_center_distance is None
            else float(np.exp(-preferred_center_distance / 0.020))
        )
        tip_distance, tip_indices = point_tree.query(tips, k=1)
        tip_heat = full_heat[tip_indices]
        center_distance, center_index = point_tree.query(contact_center, k=1)
        center_heat = float(full_heat[int(center_index)])
        contact_heat_score = float(
            0.45 * np.mean(tip_heat)
            + 0.35 * np.max(tip_heat)
            + 0.20 * center_heat
        )
        surface_score = float(np.exp(-np.mean(tip_distance) / 0.012))
        pair_distance = np.inf
        pair_alignment = 0.0
        pair_prior = 0.0
        pair_index = -1
        pair_vertical_delta = None
        if pair_tree is not None:
            pair_distance, pair_index = pair_tree.query(contact_center, k=1)
            closing_axis = row[4:13].reshape(3, 3)[:, 1]
            pair_alignment = float(abs(closing_axis @ pair_chords[int(pair_index)]))
            pair_prior = float(pair_scores[int(pair_index)])
            if desired_approach is not None:
                pair_vertical_delta = float(
                    abs(
                        (
                            full_points[opposite_indices[int(pair_index)]]
                            - full_points[visible_indices[int(pair_index)]]
                        )
                        @ desired_approach
                    )
                )
        pair_support = (
            float(np.exp(-float(pair_distance) / 0.025))
            * pair_alignment
            * pair_prior
            if np.isfinite(pair_distance)
            else 0.0
        )
        approach_angle_deg = None
        closing_angle_deg = None
        if desired_closing is not None:
            closing_axis = row[4:13].reshape(3, 3)[:, 1]
            closing_angle_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(abs(closing_axis @ desired_closing), 0.0, 1.0)
                    )
                )
            )
        if desired_approach is None:
            final_score = float(
                0.40 * score_norm[index]
                + 0.35 * contact_heat_score
                + 0.10 * surface_score
                + 0.15 * pair_support
            )
        else:
            approach_axis = row[4:13].reshape(3, 3)[:, 0]
            approach_angle_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(approach_axis @ desired_approach, -1.0, 1.0)
                    )
                )
            )
            topdown_score = float(np.exp(-approach_angle_deg / 15.0))
            if preferred_center_score is None:
                final_score = float(
                    0.30 * score_norm[index]
                    + 0.30 * contact_heat_score
                    + 0.10 * surface_score
                    + 0.15 * pair_support
                    + 0.15 * topdown_score
                )
            else:
                final_score = float(
                    0.20 * score_norm[index]
                    + 0.20 * contact_heat_score
                    + 0.10 * surface_score
                    + 0.15 * pair_support
                    + 0.15 * topdown_score
                    + 0.20 * preferred_center_score
                )
        records.append(
            {
                "candidate_index": int(index),
                "graspnet_score": float(row[0]),
                "graspnet_score_normalized": float(score_norm[index]),
                "contact_heat_score": contact_heat_score,
                "left_tip_heat": float(tip_heat[0]),
                "right_tip_heat": float(tip_heat[1]),
                "mean_tip_surface_distance_m": float(np.mean(tip_distance)),
                "center_surface_distance_m": float(center_distance),
                "nearest_pair_index": int(pair_index),
                "nearest_pair_distance_m": (
                    None if not np.isfinite(pair_distance) else float(pair_distance)
                ),
                "pair_closing_alignment": pair_alignment,
                "pair_vertical_delta_m": pair_vertical_delta,
                "pair_support": pair_support,
                "approach_to_desired_angle_deg": approach_angle_deg,
                "closing_to_desired_angle_deg": closing_angle_deg,
                "final_score": final_score,
                "width_m": float(row[1]),
                "translation_camera": row[13:16].astype(float).tolist(),
                "preferred_contact_center_distance_m": preferred_center_distance,
                "preferred_contact_center_score": preferred_center_score,
            }
        )
    order = np.argsort([record["final_score"] for record in records])[::-1]
    ranked = candidates[order]
    ranked_records = []
    for rank, source_index in enumerate(order):
        record = dict(records[int(source_index)])
        record["rank"] = int(rank)
        ranked_records.append(record)
    return ranked, ranked_records


def _refine_graspnet_candidates_with_antipodal_pairs(
    candidates: np.ndarray,
    full_points: np.ndarray,
    pair_array: np.ndarray,
    *,
    pairs_per_candidate: int,
    max_pair_distance: float,
    width_margin: float,
    max_gripper_width: float,
    desired_approach: np.ndarray | None = None,
    desired_closing: np.ndarray | None = None,
    max_approach_angle_deg: float = 180.0,
    max_closing_angle_deg: float = 180.0,
    max_pair_vertical_delta: float = float("inf"),
) -> tuple[np.ndarray, list[dict]]:
    """Snap GraspNet approach proposals onto geometry-verified contact pairs.

    GraspNet supplies learned approach directions.  The pair geometry supplies
    the closing axis, center and opening width that are unobservable in the
    original single-view heat map.
    """

    if not len(pair_array):
        return candidates[:0], []
    original_pair_indices = np.arange(len(pair_array), dtype=np.int64)
    visible_indices = pair_array[:, 0].astype(np.int64)
    opposite_indices = pair_array[:, 1].astype(np.int64)
    left = full_points[visible_indices]
    right = full_points[opposite_indices]
    centers = 0.5 * (left + right)
    chords = right - left
    widths = np.linalg.norm(chords, axis=-1)
    chords /= np.maximum(widths[:, None], 1e-8)
    pair_vertical_deltas = np.zeros(len(chords), dtype=np.float32)
    pair_topdown_angles = np.zeros(len(chords), dtype=np.float32)
    if desired_approach is not None:
        desired_approach = np.asarray(desired_approach, dtype=np.float32)
        desired_approach /= max(float(np.linalg.norm(desired_approach)), 1e-8)
        pair_vertical_deltas = np.abs((right - left) @ desired_approach)
        pair_topdown_angles = np.degrees(
            np.arcsin(np.clip(np.abs(chords @ desired_approach), 0.0, 1.0))
        ).astype(np.float32)
        feasible = (
            (pair_vertical_deltas <= max_pair_vertical_delta)
            & (pair_topdown_angles <= max_approach_angle_deg)
        )
        if desired_closing is not None:
            desired_closing = np.asarray(desired_closing, dtype=np.float32)
            desired_closing /= max(float(np.linalg.norm(desired_closing)), 1e-8)
            pair_closing_angles = np.degrees(
                np.arccos(
                    np.clip(np.abs(chords @ desired_closing), 0.0, 1.0)
                )
            )
            feasible &= pair_closing_angles <= max_closing_angle_deg
        original_pair_indices = original_pair_indices[feasible]
        left = left[feasible]
        right = right[feasible]
        centers = centers[feasible]
        chords = chords[feasible]
        widths = widths[feasible]
        pair_vertical_deltas = pair_vertical_deltas[feasible]
        pair_topdown_angles = pair_topdown_angles[feasible]
        if not len(centers):
            return candidates[:0], []
    pair_tree = cKDTree(centers)
    refined_rows = []
    records = []
    query_k = min(max(1, pairs_per_candidate), len(centers))
    for candidate_index, source in enumerate(candidates):
        distances, pair_indices = pair_tree.query(
            source[13:16],
            k=query_k,
        )
        distances = np.atleast_1d(distances)
        pair_indices = np.atleast_1d(pair_indices)
        source_rotation = source[4:13].reshape(3, 3)
        source_approach = source_rotation[:, 0]
        source_closing = source_rotation[:, 1]
        for distance, pair_index in zip(distances, pair_indices, strict=True):
            if float(distance) > max_pair_distance:
                continue
            pair_index = int(pair_index)
            original_pair_index = int(original_pair_indices[pair_index])
            closing = chords[pair_index].copy()
            if float(closing @ source_closing) < 0:
                closing *= -1
            closing_alignment = float(abs(closing @ source_closing))
            approach_seed = (
                source_approach
                if desired_approach is None
                else desired_approach
            )
            approach = approach_seed - closing * float(approach_seed @ closing)
            approach_norm = float(np.linalg.norm(approach))
            if approach_norm < 1e-5:
                continue
            approach /= approach_norm
            vertical = np.cross(approach, closing)
            vertical_norm = float(np.linalg.norm(vertical))
            if vertical_norm < 1e-5:
                continue
            vertical /= vertical_norm
            closing = np.cross(vertical, approach)
            closing /= max(float(np.linalg.norm(closing)), 1e-8)
            rotation = np.stack((approach, closing, vertical), axis=-1)

            pair_width = float(widths[pair_index])
            opening = min(pair_width + width_margin, max_gripper_width)
            if opening <= pair_width:
                continue
            depth = float(np.clip(source[3], 0.015, 0.040))
            # In GraspNet coordinates local +X runs from the gripper base to
            # the finger tips, so place the origin one depth behind contacts.
            translation = centers[pair_index] - approach * depth
            pair_score = float(pair_array[original_pair_index, 2])
            topdown_alignment = (
                1.0
                if desired_approach is None
                else float(np.clip(approach @ desired_approach, 0.0, 1.0))
            )
            refinement_score = float(
                source[0]
                * pair_score
                * np.exp(-float(distance) / 0.030)
                * (0.5 + 0.5 * closing_alignment)
                * topdown_alignment
            )
            row = source.copy()
            row[0] = refinement_score
            row[1] = opening
            row[3] = depth
            row[4:13] = rotation.reshape(-1)
            row[13:16] = translation
            refined_rows.append(row)
            records.append(
                {
                    "source_candidate_index": int(candidate_index),
                    "pair_index": original_pair_index,
                    "source_graspnet_score": float(source[0]),
                    "refinement_score": refinement_score,
                    "pair_score": pair_score,
                    "pair_distance_m": float(distance),
                    "pair_width_m": pair_width,
                    "refined_width_m": opening,
                    "closing_alignment_before_refine": closing_alignment,
                    "pair_vertical_delta_m": float(
                        pair_vertical_deltas[pair_index]
                    ),
                    "topdown_approach_angle_deg": (
                        None
                        if desired_approach is None
                        else float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        approach @ desired_approach,
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                    ),
                }
            )
    if not refined_rows:
        return candidates[:0], []
    order = np.argsort([row[0] for row in refined_rows])[::-1]
    return (
        np.asarray(refined_rows, dtype=np.float32)[order],
        [records[int(index)] for index in order],
    )


def _save_open3d(
    output_path: Path,
    snapshot: dict[str, np.ndarray],
    full_points: np.ndarray,
    full_heat: np.ndarray,
    candidates: np.ndarray,
    *,
    top_k: int,
) -> None:
    import open3d as o3d

    # Render in the simulator camera convention and crop around the complete
    # task object. The hot manipulated part is drawn over the collision scene.
    height = width = 800
    focus_points = (
        np.asarray(snapshot["complete_scene_points_camera"], dtype=np.float32)
        if "complete_scene_points_camera" in snapshot
        else full_points
    )
    render_offset = np.array(
        [
            0.5
            * (
                np.quantile(focus_points[:, 0], 0.005)
                + np.quantile(focus_points[:, 0], 0.995)
            ),
            0.5
            * (
                np.quantile(focus_points[:, 1], 0.005)
                + np.quantile(focus_points[:, 1], 0.995)
            ),
            0.0,
        ],
        dtype=np.float32,
    )
    render_points = full_points - render_offset[None]
    render_focus = focus_points - render_offset[None]
    normalized_xy = render_focus[:, :2] / np.maximum(
        render_focus[:, 2:3],
        1e-6,
    )
    lower = np.quantile(normalized_xy, 0.005, axis=0)
    upper = np.quantile(normalized_xy, 0.995, axis=0)
    extent = np.maximum(upper - lower, 1e-4)
    focal = float(min(0.70 * width / extent[0], 0.70 * height / extent[1]))

    heat_scale = max(float(np.quantile(full_heat, 0.995)), 1e-6)
    normalized_heat = np.clip(full_heat / heat_scale, 0.0, 1.0)
    heat_u8 = np.round(normalized_heat * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8[:, None], cv2.COLORMAP_TURBO)[:, 0]
    turbo_rgb = heat_bgr[:, ::-1].astype(np.float64) / 255.0
    alpha = np.clip(np.sqrt(normalized_heat)[:, None], 0.08, 1.0)
    heat_rgb = (1.0 - alpha) * 0.58 + alpha * turbo_rgb
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(render_points)
    cloud.colors = o3d.utility.Vector3dVector(heat_rgb)

    geometries = []
    if "complete_scene_points_camera" in snapshot:
        context_points = np.asarray(
            snapshot["complete_scene_points_camera"], dtype=np.float32
        )
        context_colors = np.asarray(
            snapshot["complete_scene_colors"], dtype=np.float64
        )
        context_is_manipulated = np.asarray(
            snapshot["complete_scene_is_manipulated"], dtype=bool
        )
        context = o3d.geometry.PointCloud()
        context.points = o3d.utility.Vector3dVector(
            context_points[~context_is_manipulated] - render_offset[None]
        )
        context.colors = o3d.utility.Vector3dVector(
            np.clip(0.55 * context_colors[~context_is_manipulated] + 0.30, 0, 1)
        )
        geometries.append(context)
    geometries.append(cloud)
    colors = [
        np.array([1.0, 0.1, 0.1]) if index == 0 else np.array([0.1, 1.0, 0.2])
        for index in range(min(top_k, len(candidates)))
    ]
    for candidate_index, (row, color) in enumerate(
        zip(candidates[:top_k], colors, strict=True)
    ):
        points = _grasp_points(row) - render_offset[None]
        radius = 0.0024 if candidate_index == 0 else 0.0014
        for start_index, end_index in ((0, 1), (2, 3), (0, 2), (4, 5)):
            start = points[start_index]
            end = points[end_index]
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1e-8:
                continue
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius,
                height=length,
                resolution=16,
            )
            cylinder.compute_vertex_normals()
            cylinder.paint_uniform_color(color)
            align = np.eye(4, dtype=np.float64)
            direction = delta / length
            cross = np.cross(np.array([0.0, 0.0, 1.0]), direction)
            cross_norm = float(np.linalg.norm(cross))
            cosine = float(np.clip(direction[2], -1.0, 1.0))
            if cross_norm < 1e-8:
                axis_angle = (
                    np.zeros(3, dtype=np.float64)
                    if cosine >= 0
                    else np.array([np.pi, 0.0, 0.0], dtype=np.float64)
                )
            else:
                axis_angle = cross / cross_norm * np.arccos(cosine)
            align[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis_angle
            )
            align[:3, 3] = 0.5 * (start + end)
            cylinder.transform(align)
            geometries.append(cylinder)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=width, height=height, visible=False)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    options = visualizer.get_render_option()
    options.background_color = np.array([0.97, 0.97, 0.97])
    options.point_size = 5.0
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        focal,
        focal,
        width / 2,
        height / 2,
    )
    camera.extrinsic = np.eye(4, dtype=np.float64)
    visualizer.get_view_control().convert_from_pinhole_camera_parameters(
        camera,
        allow_arbitrary=True,
    )
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(str(output_path), do_render=True)
    visualizer.destroy_window()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GraspNet on the full ManiSkill mug cloud, constrained by propagated contact heat."
    )
    parser.add_argument(
        "--input",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0/contact_propagation.npz"
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0/pouring_snapshot.npz"
        ),
    )
    parser.add_argument("--graspnet-root", default="/home/users1/ljian/graspnet-baseline")
    parser.add_argument("--checkpoint", default="checkpoint-rs.tar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=25600)
    parser.add_argument("--target-ratio", type=float, default=0.70)
    parser.add_argument("--heat-threshold", type=float, default=0.10)
    parser.add_argument("--collision-voxel", type=float, default=0.008)
    parser.add_argument("--collision-threshold", type=float, default=0.10)
    parser.add_argument("--max-global-collision-iou", type=float, default=0.02)
    parser.add_argument("--max-finger-collision-iou", type=float, default=0.02)
    parser.add_argument("--max-palm-collision-iou", type=float, default=0.01)
    parser.add_argument("--max-approach-collision-iou", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument(
        "--max-decoded-before-refine",
        type=int,
        default=150,
        help="Top GraspNet proposals retained before antipodal expansion.",
    )
    parser.add_argument(
        "--max-refined-before-collision",
        type=int,
        default=600,
        help="Top refined proposals retained before dense collision matrices.",
    )
    parser.add_argument("--pairs-per-candidate", type=int, default=24)
    parser.add_argument("--max-pair-distance", type=float, default=0.100)
    parser.add_argument("--width-margin", type=float, default=0.006)
    parser.add_argument("--max-gripper-width", type=float, default=0.080)
    parser.add_argument("--no-antipodal-refine", action="store_true")
    parser.add_argument(
        "--no-grasp-nms",
        action="store_true",
        help="Keep nearby contact-centre alternatives for task-specific filtering.",
    )
    parser.add_argument(
        "--no-hard-workspace-mask",
        action="store_true",
        help=(
            "Do not suppress GraspNet seeds outside the hot region. Use when "
            "a small contact ROI should filter/refine full-object proposals "
            "instead of replacing the geometric context seen by GraspNet."
        ),
    )
    parser.add_argument(
        "--topdown",
        action="store_true",
        help="Require approach to be as close as possible to world -Z.",
    )
    parser.add_argument(
        "--desired-approach-world",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Task-specific gripper approach axis in world coordinates. "
            "For a drawer this points from free space toward the handle; "
            "--topdown remains a backward-compatible shorthand for [0,0,-1]."
        ),
    )
    parser.add_argument(
        "--desired-closing-world",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Preferred unsigned parallel-jaw closing axis in world coordinates.",
    )
    parser.add_argument("--closing-max-angle-deg", type=float, default=35.0)
    parser.add_argument(
        "--topdown-max-angle-deg",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--pair-max-vertical-delta",
        type=float,
        default=0.003,
        help="Maximum world-Z difference between the two contact points.",
    )
    parser.add_argument(
        "--min-pair-closing-alignment",
        type=float,
        default=0.0,
        help=(
            "Minimum unsigned dot product between the verified antipodal "
            "contact chord and the gripper closing axis."
        ),
    )
    parser.add_argument(
        "--max-nearest-pair-distance",
        type=float,
        default=float("inf"),
        help=(
            "Maximum distance in metres from the fingertip midpoint to the "
            "centre of its nearest verified antipodal contact pair."
        ),
    )
    parser.add_argument(
        "--min-both-tip-heat",
        type=float,
        default=0.0,
        help="Hard lower bound applied independently to both fingertips.",
    )
    parser.add_argument(
        "--max-tip-surface-distance",
        type=float,
        default=0.010,
    )
    parser.add_argument("--visualize-top-k", type=int, default=5)
    parser.add_argument(
        "--preferred-contact-center-object",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Preferred contact centre in the manipulated-part local frame.",
    )
    parser.add_argument(
        "--max-preferred-center-distance",
        type=float,
        default=float("inf"),
    )
    parser.add_argument(
        "--snap-preferred-contact-center",
        action="store_true",
        help=(
            "Augment GraspNet orientations with translations whose fingertip "
            "midpoint is snapped to the preferred manipulated-part centre; "
            "strict contact and collision filters are still applied afterward."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent
    propagation = np.load(input_path)
    snapshot = np.load(args.snapshot)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    full_points = np.asarray(propagation["full_points_camera"], dtype=np.float32)
    full_heat = np.asarray(propagation["full_heat"], dtype=np.float32)
    scene_points = np.asarray(propagation["scene_points_camera"], dtype=np.float32)
    scene_colors = np.asarray(propagation["scene_colors"], dtype=np.float32)
    if float(scene_colors.max(initial=0.0)) > 1.0:
        scene_colors /= 255.0
    scene_is_manipulated_key = (
        "scene_is_manipulated"
        if "scene_is_manipulated" in propagation.files
        else "scene_is_cup"
    )
    scene_is_manipulated = np.asarray(
        propagation[scene_is_manipulated_key], dtype=bool
    )
    scene_background_points = scene_points[~scene_is_manipulated]
    scene_background_colors = scene_colors[~scene_is_manipulated]
    visible_color_key = (
        "manipulated_colors"
        if "manipulated_colors" in snapshot.files
        else "visible_colors"
    )
    visible_colors = np.asarray(snapshot[visible_color_key], dtype=np.float32)
    if float(visible_colors.max(initial=0.0)) > 1.0:
        visible_colors /= 255.0
    full_color = np.median(visible_colors, axis=0)
    desired_approach_camera = None
    desired_closing_camera = None
    desired_approach_world = None
    if args.desired_approach_world is not None:
        desired_approach_world = np.asarray(
            args.desired_approach_world, dtype=np.float32
        )
    elif args.topdown:
        desired_approach_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    if desired_approach_world is not None:
        desired_approach_world /= max(
            float(np.linalg.norm(desired_approach_world)), 1e-8
        )
        desired_approach_camera = (
            np.asarray(snapshot["T_world_to_camera"], dtype=np.float32)[:3, :3]
            @ desired_approach_world
        )
        desired_approach_camera /= max(
            float(np.linalg.norm(desired_approach_camera)),
            1e-8,
        )
    if args.desired_closing_world is not None:
        desired_closing_world = np.asarray(
            args.desired_closing_world, dtype=np.float32
        )
        desired_closing_world /= max(
            float(np.linalg.norm(desired_closing_world)), 1e-8
        )
        desired_closing_camera = (
            np.asarray(snapshot["T_world_to_camera"], dtype=np.float32)[:3, :3]
            @ desired_closing_world
        )
        desired_closing_camera /= max(
            float(np.linalg.norm(desired_closing_camera)), 1e-8
        )

    endpoints, collision_points, _colors, sampling_debug = _compose_graspnet_input(
        full_points,
        full_heat,
        full_color,
        scene_background_points,
        scene_background_colors,
        num_points=args.num_points,
        target_ratio=args.target_ratio,
        heat_threshold=args.heat_threshold,
        rng=rng,
    )

    graspnet_root = Path(args.graspnet_root)
    GraspNet, pred_decode, GraspGroup, ModelFreeCollisionDetector = (
        _install_graspnet_imports(graspnet_root)
    )
    network = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).cuda()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = graspnet_root / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    with torch.no_grad():
        decoded = network(endpoints)
        seed_mask = torch.gather(
            decoded["workspace_mask"],
            1,
            decoded["fp2_inds"].long(),
        )
        if not args.no_hard_workspace_mask:
            if "objectness_score" in decoded:
                decoded["objectness_score"][:, 1, :] -= 1e5 * (1.0 - seed_mask)
            if "graspness_score" in decoded:
                decoded["graspness_score"] *= seed_mask
        raw_predictions = pred_decode(decoded)[0].detach().cpu().numpy()

    num_decoded = int(len(raw_predictions))
    np.save(output_dir / "graspnet_decoded_before_refine.npy", raw_predictions)
    if not num_decoded:
        raise RuntimeError(
            "GraspNet decoded no grasp from the heat-constrained full cloud; "
            "lower --heat-threshold or increase --target-ratio"
        )
    decoded_order = np.argsort(raw_predictions[:, 0])[::-1]
    raw_for_refinement = raw_predictions[
        decoded_order[: args.max_decoded_before_refine]
    ]
    pair_array = np.asarray(propagation["antipodal_pairs"], dtype=np.float32)
    refinement_records = []
    if args.no_antipodal_refine:
        grasp_predictions = raw_for_refinement
    else:
        grasp_predictions, refinement_records = (
            _refine_graspnet_candidates_with_antipodal_pairs(
                raw_for_refinement,
                full_points,
                pair_array,
                pairs_per_candidate=args.pairs_per_candidate,
                max_pair_distance=args.max_pair_distance,
                width_margin=args.width_margin,
                max_gripper_width=args.max_gripper_width,
                desired_approach=desired_approach_camera,
                desired_closing=desired_closing_camera,
                max_approach_angle_deg=args.topdown_max_angle_deg,
                max_closing_angle_deg=args.closing_max_angle_deg,
                max_pair_vertical_delta=args.pair_max_vertical_delta,
            )
        )
    if not len(grasp_predictions):
        raise RuntimeError(
            "No GraspNet proposal could be paired with contacts under the "
            "requested approach and vertical-pair constraints"
        )
    num_refined_generated = int(len(grasp_predictions))
    grasp_predictions = grasp_predictions[: args.max_refined_before_collision]
    refinement_records = refinement_records[: args.max_refined_before_collision]
    np.save(output_dir / "graspnet_refined_before_collision.npy", grasp_predictions)
    grasp_group = GraspGroup(grasp_predictions)
    detector = ModelFreeCollisionDetector(
        collision_points,
        voxel_size=args.collision_voxel,
    )
    collision, prefilter_ious = detector.detect(
        grasp_group,
        approach_dist=0.01,
        collision_thresh=args.collision_threshold,
        return_ious=True,
    )
    grasp_group = grasp_group[~collision]
    num_collision_free = int(len(grasp_group))
    if not len(grasp_group):
        raise RuntimeError("GraspNet generated no collision-free grasp on the full cloud")
    if not args.no_grasp_nms:
        grasp_group = grasp_group.nms()
    grasp_group.sort_by_score()
    candidates = grasp_group.grasp_group_array[: args.max_candidates].astype(np.float32)
    preferred_contact_center_camera = None
    if args.preferred_contact_center_object is not None:
        preferred_object = np.asarray(
            args.preferred_contact_center_object, dtype=np.float32
        )
        T_object_to_camera = np.asarray(
            snapshot["T_manipulated_to_camera"], dtype=np.float32
        )
        preferred_contact_center_camera = (
            T_object_to_camera[:3, :3] @ preferred_object
            + T_object_to_camera[:3, 3]
        )
    if (
        args.snap_preferred_contact_center
        and preferred_contact_center_camera is not None
        and len(candidates)
    ):
        snapped = candidates.copy()
        for row in snapped:
            current_center = _grasp_points(row)[[1, 3]].mean(axis=0)
            row[13:16] += preferred_contact_center_camera - current_center
        candidates = np.concatenate((candidates, snapped), axis=0)
    ranked, rank_records = _rank_by_contact(
        candidates,
        full_points,
        full_heat,
        pair_array,
        desired_approach=desired_approach_camera,
        desired_closing=desired_closing_camera,
        preferred_contact_center=preferred_contact_center_camera,
    )
    if desired_approach_camera is not None:
        constrained_rows = []
        constrained_records = []
        constraint_diagnostics = []
        constraint_pass_counts = {
            "total": len(rank_records),
            "approach": 0,
            "closing": 0,
            "pair_vertical": 0,
            "pair_closing_alignment": 0,
            "nearest_pair_distance": 0,
            "left_heat": 0,
            "right_heat": 0,
            "tip_surface": 0,
            "preferred_center": 0,
        }
        for row, record in zip(ranked, rank_records, strict=True):
            checks = {
                "approach": record["approach_to_desired_angle_deg"]
                <= args.topdown_max_angle_deg,
                "closing": desired_closing_camera is None
                or record["closing_to_desired_angle_deg"]
                <= args.closing_max_angle_deg,
                "pair_vertical": record["pair_vertical_delta_m"]
                <= args.pair_max_vertical_delta,
                "pair_closing_alignment": record["pair_closing_alignment"]
                >= args.min_pair_closing_alignment,
                "nearest_pair_distance": record["nearest_pair_distance_m"]
                is not None
                and record["nearest_pair_distance_m"]
                <= args.max_nearest_pair_distance,
                "left_heat": record["left_tip_heat"] >= args.min_both_tip_heat,
                "right_heat": record["right_tip_heat"] >= args.min_both_tip_heat,
                "tip_surface": record["mean_tip_surface_distance_m"]
                <= args.max_tip_surface_distance,
                "preferred_center": record["preferred_contact_center_distance_m"]
                is None
                or record["preferred_contact_center_distance_m"]
                <= args.max_preferred_center_distance,
            }
            for name, passed in checks.items():
                constraint_pass_counts[name] += int(passed)
            constraint_diagnostics.append(
                {
                    "rank_before_constraints": int(record["rank"]),
                    "candidate_index": int(record["candidate_index"]),
                    "checks": {name: bool(passed) for name, passed in checks.items()},
                    "approach_to_desired_angle_deg": record[
                        "approach_to_desired_angle_deg"
                    ],
                    "closing_to_desired_angle_deg": record[
                        "closing_to_desired_angle_deg"
                    ],
                    "pair_vertical_delta_m": record["pair_vertical_delta_m"],
                    "pair_closing_alignment": record["pair_closing_alignment"],
                    "nearest_pair_distance_m": record["nearest_pair_distance_m"],
                    "preferred_contact_center_distance_m": record[
                        "preferred_contact_center_distance_m"
                    ],
                    "left_tip_heat": record["left_tip_heat"],
                    "right_tip_heat": record["right_tip_heat"],
                    "mean_tip_surface_distance_m": record[
                        "mean_tip_surface_distance_m"
                    ],
                }
            )
            if (
                all(checks.values())
            ):
                constrained_rows.append(row)
                constrained_records.append(record)
        ranked = np.asarray(constrained_rows, dtype=np.float32).reshape(-1, 17)
        rank_records = constrained_records
        for rank, record in enumerate(rank_records):
            record["rank"] = rank
    if not len(ranked):
        diagnostic_path = output_dir / "grasp_constraint_diagnostics.json"
        diagnostic_path.write_text(
            json.dumps(
                {
                    "pass_counts": constraint_pass_counts,
                    "candidates": constraint_diagnostics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"constraint_pass_counts": constraint_pass_counts}, indent=2))
        raise RuntimeError(
            "No GraspNet candidates remained after contact/approach hard constraints"
        )

    collision_part_names = (
        "global_iou",
        "left_finger_iou",
        "right_finger_iou",
        "palm_iou",
        "approach_path_iou",
    )
    num_ranked_before_strict_collision = int(len(ranked))
    ranked_collision, ranked_ious = detector.detect(
        GraspGroup(ranked),
        approach_dist=0.01,
        collision_thresh=args.collision_threshold,
        return_ious=True,
    )
    strict_collision_limits = CollisionLimits(
        global_iou=args.max_global_collision_iou,
        finger_iou=args.max_finger_collision_iou,
        palm_iou=args.max_palm_collision_iou,
        approach_path_iou=args.max_approach_collision_iou,
    )
    strict_collision_keep = strict_collision_mask(
        ranked_collision,
        ranked_ious,
        strict_collision_limits,
    )
    np.save(output_dir / "graspnet_ranked_before_strict_collision.npy", ranked)
    strict_collision_records = []
    for index, record in enumerate(rank_records):
        strict_collision_records.append(
            {
                "rank_before_strict_collision": int(index),
                "translation_camera": np.asarray(ranked[index, 13:16])
                .astype(float)
                .tolist(),
                "approach_to_desired_angle_deg": record[
                    "approach_to_desired_angle_deg"
                ],
                "closing_to_desired_angle_deg": record[
                    "closing_to_desired_angle_deg"
                ],
                "left_tip_heat": record["left_tip_heat"],
                "right_tip_heat": record["right_tip_heat"],
                "mean_tip_surface_distance_m": record[
                    "mean_tip_surface_distance_m"
                ],
                "collision_part_ious": {
                    name: float(values[index])
                    for name, values in zip(
                        collision_part_names, ranked_ious, strict=True
                    )
                },
                "detector_collision": bool(ranked_collision[index]),
                "strict_collision_pass": bool(strict_collision_keep[index]),
            }
        )
    strict_diagnostics_path = output_dir / "graspnet_strict_collision_diagnostics.json"
    strict_diagnostics_path.write_text(
        json.dumps(
            {
                "limits": strict_collision_limits.as_dict(),
                "num_candidates": int(len(ranked)),
                "num_passing": int(strict_collision_keep.sum()),
                "candidates": strict_collision_records[:100],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    strictly_collision_free_rows = []
    strictly_collision_free_records = []
    for index, (row, record) in enumerate(zip(ranked, rank_records, strict=True)):
        if not strict_collision_keep[index]:
            continue
        record = dict(record)
        record["collision_part_ious"] = {
            name: float(values[index])
            for name, values in zip(collision_part_names, ranked_ious, strict=True)
        }
        strictly_collision_free_rows.append(row)
        strictly_collision_free_records.append(record)
    ranked = np.asarray(strictly_collision_free_rows, dtype=np.float32).reshape(-1, 17)
    rank_records = strictly_collision_free_records
    for rank, record in enumerate(rank_records):
        record["rank"] = rank
    if not len(ranked):
        raise RuntimeError(
            "No grasp remained after strict per-part collision constraints; "
            f"inspect {strict_diagnostics_path} before relaxing thresholds"
        )
    selected_collision = rank_records[0]["collision_part_ious"]
    prefilter_collision_summary = {
        name: {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        for name, values in zip(
            collision_part_names,
            prefilter_ious,
            strict=True,
        )
    }

    selected_camera = ranked[0]
    T_camera_to_world = np.linalg.inv(snapshot["T_world_to_camera"]).astype(
        np.float32
    )
    object_transform_key = (
        "T_manipulated_to_camera"
        if "T_manipulated_to_camera" in snapshot.files
        else "T_object_to_camera"
    )
    T_camera_to_object = np.linalg.inv(snapshot[object_transform_key]).astype(
        np.float32
    )
    selected_world = _transform_grasp_row(selected_camera, T_camera_to_world)
    selected_object = _transform_grasp_row(selected_camera, T_camera_to_object)
    rank_records[0]["translation_world"] = selected_world[13:16].astype(float).tolist()
    rank_records[0]["translation_object"] = (
        selected_object[13:16].astype(float).tolist()
    )
    selected_pair_index = int(rank_records[0]["nearest_pair_index"])
    if selected_pair_index >= 0:
        selected_pair_indices = pair_array[selected_pair_index, :2].astype(
            np.int64
        )
        selected_pair_camera = full_points[selected_pair_indices]

        def transform_pair(transform: np.ndarray) -> np.ndarray:
            return (
                selected_pair_camera @ transform[:3, :3].T
                + transform[:3, 3]
            )

        selected_pair_world = transform_pair(T_camera_to_world)
        selected_pair_object = transform_pair(T_camera_to_object)
        rank_records[0]["contact_pair_point_indices"] = (
            selected_pair_indices.astype(int).tolist()
        )
        rank_records[0]["contact_pair_width_m"] = float(
            np.linalg.norm(selected_pair_camera[1] - selected_pair_camera[0])
        )
        rank_records[0]["contact_pair_geometry"] = {
            "score": float(pair_array[selected_pair_index, 2]),
            "width_m": float(pair_array[selected_pair_index, 3]),
            "visible_alignment": float(pair_array[selected_pair_index, 4]),
            "opposite_alignment": float(pair_array[selected_pair_index, 5]),
            "normal_opposition": float(pair_array[selected_pair_index, 6]),
            "opposite_visible_distance_m": float(
                pair_array[selected_pair_index, 7]
            ),
            "opposite_is_hidden": bool(pair_array[selected_pair_index, 8]),
        }
        rank_records[0]["contact_pair_camera"] = (
            selected_pair_camera.astype(float).tolist()
        )
        rank_records[0]["contact_pair_world"] = (
            selected_pair_world.astype(float).tolist()
        )
        rank_records[0]["contact_pair_object"] = (
            selected_pair_object.astype(float).tolist()
        )

    np.save(output_dir / "graspnet_full_candidates_raw.npy", candidates)
    np.save(output_dir / "graspnet_full_candidates_ranked.npy", ranked)
    np.save(output_dir / "graspnet_selected.npy", selected_camera)
    np.save(output_dir / "graspnet_selected_world.npy", selected_world)
    np.save(output_dir / "graspnet_selected_object.npy", selected_object)

    canvas = cv2.cvtColor(snapshot["rgb"], cv2.COLOR_RGB2BGR)
    for index in range(min(args.visualize_top_k, len(ranked)) - 1, -1, -1):
        _draw_grasp(
            canvas,
            ranked[index],
            snapshot["intrinsic_cv"],
            (0, 0, 255) if index == 0 else (30, 220, 60),
            3 if index == 0 else 1,
            f"{index}:{rank_records[index]['final_score']:.2f}",
        )
    cv2.imwrite(str(output_dir / "graspnet_selected_rgb.png"), canvas)
    selected_canvas = cv2.cvtColor(snapshot["rgb"], cv2.COLOR_RGB2BGR)
    _draw_grasp(
        selected_canvas,
        ranked[0],
        snapshot["intrinsic_cv"],
        (0, 0, 255),
        3,
        f"selected:{rank_records[0]['final_score']:.2f}",
    )
    cv2.imwrite(
        str(output_dir / "graspnet_selected_rgb_clean.png"), selected_canvas
    )
    _save_open3d(
        output_dir / "graspnet_full_heat_open3d.png",
        snapshot,
        full_points,
        full_heat,
        ranked,
        top_k=args.visualize_top_k,
    )
    _save_open3d(
        output_dir / "graspnet_selected_open3d.png",
        snapshot,
        full_points,
        full_heat,
        ranked,
        top_k=1,
    )

    report = {
        "graspnet_checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "sampling": sampling_debug,
        "num_decoded_grasps": num_decoded,
        "num_decoded_retained_before_refine": int(len(raw_for_refinement)),
        "num_antipodal_refined_grasps_generated": num_refined_generated,
        "num_antipodal_refined_grasps": int(len(grasp_predictions)),
        "num_collision_free_grasps": num_collision_free,
        "num_ranked_before_strict_collision": num_ranked_before_strict_collision,
        "num_ranked_grasps": int(len(ranked)),
        "collision": {
            "voxel_size": args.collision_voxel,
            "threshold": args.collision_threshold,
            "approach_distance": 0.01,
            "strict_part_thresholds": strict_collision_limits.as_dict(),
            "selected_part_ious": selected_collision,
            "prefilter_part_iou_summary": prefilter_collision_summary,
        },
        "selected": rank_records[0],
        "antipodal_refinement_enabled": not args.no_antipodal_refine,
        "hard_workspace_mask_enabled": not args.no_hard_workspace_mask,
        "approach_constraint": {
            "enabled": desired_approach_camera is not None,
            "mode": "topdown" if args.topdown else "task_axis",
            "desired_world_axis": (
                None
                if desired_approach_world is None
                else desired_approach_world.astype(float).tolist()
            ),
            "desired_camera_axis": (
                None
                if desired_approach_camera is None
                else desired_approach_camera.astype(float).tolist()
            ),
            "desired_closing_world_axis": args.desired_closing_world,
            "desired_closing_camera_axis": (
                None
                if desired_closing_camera is None
                else desired_closing_camera.astype(float).tolist()
            ),
            "max_closing_angle_deg": args.closing_max_angle_deg,
            "max_approach_angle_deg": args.topdown_max_angle_deg,
            "max_pair_vertical_delta_m": args.pair_max_vertical_delta,
            "min_pair_closing_alignment": args.min_pair_closing_alignment,
            "max_nearest_pair_distance_m": args.max_nearest_pair_distance,
            "min_both_tip_heat": args.min_both_tip_heat,
            "max_tip_surface_distance_m": args.max_tip_surface_distance,
            "preferred_contact_center_object": args.preferred_contact_center_object,
            "max_preferred_center_distance_m": args.max_preferred_center_distance,
            "snap_preferred_contact_center": args.snap_preferred_contact_center,
        },
        "top_candidates": rank_records[: min(20, len(rank_records))],
        "outputs": {
            "selected_row": str(output_dir / "graspnet_selected.npy"),
            "selected_row_world": str(
                output_dir / "graspnet_selected_world.npy"
            ),
            "selected_row_object": str(
                output_dir / "graspnet_selected_object.npy"
            ),
            "rgb": str(output_dir / "graspnet_selected_rgb.png"),
            "rgb_selected_only": str(
                output_dir / "graspnet_selected_rgb_clean.png"
            ),
            "open3d": str(output_dir / "graspnet_full_heat_open3d.png"),
            "open3d_selected_only": str(
                output_dir / "graspnet_selected_open3d.png"
            ),
            "strict_collision_diagnostics": str(strict_diagnostics_path),
        },
    }
    (output_dir / "graspnet_full_contact_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
