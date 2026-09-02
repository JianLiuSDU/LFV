#!/usr/bin/env python3
"""Run the Stage 2 hierarchy on an LFV ManiSkill RGB-D snapshot.

The adapter deliberately uses the same mask -> joint pixel/XYZ/DINO sampling
path as cache construction.  Its output NPZ follows the historical execution
contract so the grasp executor can consume a new model without task-specific
changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lfv.datasets.functional_motion.cache_builder import (
    extract_dino_grid,
    sample_dense_features,
)
from lfv.datasets.functional_motion.sampling import (
    assert_unique_aligned,
    farthest_pixel_sample,
    unproject_pixels,
    valid_mask_pixels,
)
from lfv.features import DinoV2DenseExtractor
from lfv.geometry import local_delta_to_camera, pose9d_to_matrix_np
from lfv.inference.functional_motion import camera_delta_to_world_delta
from lfv.models.functional_motion_generation import load_stage2_checkpoint


def _first_existing(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in data:
            return key
    raise KeyError(f"Snapshot contains none of {keys}")


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        (
            points[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2],
            points[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2],
        ),
        axis=-1,
    ).astype(np.int32)


def _rotation_angle(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix[..., :3, :3], axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(first[..., :3, :3], -1, -2) @ second[..., :3, :3]
    return _rotation_angle(relative)


def _load_training_prior(cache_root: Path) -> dict[str, np.ndarray | float]:
    split = json.loads((cache_root / "split_manifest.json").read_text(encoding="utf-8"))
    residuals, angles, step_lengths, first_step_lengths = [], [], [], []
    for episode_id in split["episodes"]["train"]:
        with np.load(cache_root / "episodes" / f"{episode_id}.npz") as data:
            goal = np.asarray(data["goal_pose9d"], dtype=np.float32)
            reference_center = np.asarray(data["reference_points"], dtype=np.float32).mean(0)
            trajectory = pose9d_to_matrix_np(data["trajectory_pose9d"])
        residuals.append(goal[:3] - reference_center)
        angles.append(float(_rotation_angle(pose9d_to_matrix_np(goal))))
        episode_steps = np.linalg.norm(
            np.diff(trajectory[:, :3, 3], axis=0), axis=-1
        )
        step_lengths.extend(episode_steps.tolist())
        first_step_lengths.append(float(episode_steps[0]))
    residuals_array = np.asarray(residuals, dtype=np.float32)
    angles_array = np.asarray(angles, dtype=np.float32)
    return {
        "residual_mean": residuals_array.mean(0),
        "residual_std": np.maximum(residuals_array.std(0), 0.02),
        "angle_mean": float(angles_array.mean()),
        "angle_std": float(max(angles_array.std(), np.deg2rad(10.0))),
        "step_p95": float(np.quantile(np.asarray(step_lengths), 0.95)),
        "first_step_p95": float(
            np.quantile(np.asarray(first_step_lengths, dtype=np.float32), 0.95)
        ),
    }


def _rank_candidates(
    goals: np.ndarray,
    trajectories: np.ndarray,
    reference_center: np.ndarray,
    prior: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Rank generative samples without using simulation ground truth."""

    rows: list[dict] = []
    for goal_index in range(goals.shape[0]):
        goal_matrix = pose9d_to_matrix_np(goals[goal_index])
        residual = goals[goal_index, :3] - reference_center
        residual_z = (residual - prior["residual_mean"]) / prior["residual_std"]
        angle = float(_rotation_angle(goal_matrix))
        angle_z = (angle - prior["angle_mean"]) / prior["angle_std"]
        for trajectory_index in range(trajectories.shape[1]):
            matrices = pose9d_to_matrix_np(trajectories[goal_index, trajectory_index])
            endpoint_translation = float(
                np.linalg.norm(matrices[-1, :3, 3] - goal_matrix[:3, 3])
            )
            endpoint_rotation = float(_rotation_distance(matrices[-1], goal_matrix))
            steps = np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=-1)
            accelerations = np.diff(matrices[:, :3, 3], n=2, axis=0)
            smoothness = float(np.mean(np.linalg.norm(accelerations, axis=-1)))
            excessive_step = float(
                np.maximum(steps - 2.0 * prior["step_p95"], 0.0).mean()
            )
            first_step = float(steps[0])
            excessive_first_step = float(
                max(first_step - 2.0 * prior["first_step_p95"], 0.0)
            )
            score = (
                float(np.mean(residual_z**2))
                + 0.20 * float(angle_z**2)
                + 8.0 * endpoint_translation
                + 0.20 * endpoint_rotation
                + 30.0 * smoothness
                + 50.0 * excessive_step
                + 100.0 * excessive_first_step
            )
            rows.append(
                {
                    "goal_index": goal_index,
                    "trajectory_index": trajectory_index,
                    "score": score,
                    "goal_residual_z2": float(np.mean(residual_z**2)),
                    "goal_rotation_deg": float(np.rad2deg(angle)),
                    "endpoint_translation_to_goal_m": endpoint_translation,
                    "endpoint_rotation_to_goal_deg": float(np.rad2deg(endpoint_rotation)),
                    "mean_second_difference_m": smoothness,
                    "first_step_translation_m": first_step,
                    "training_first_step_p95_m": float(prior["first_step_p95"]),
                    "max_step_translation_m": float(steps.max()),
                }
            )
    rows.sort(key=lambda row: row["score"])
    selected = np.asarray(
        [rows[0]["goal_index"], rows[0]["trajectory_index"]], dtype=np.int64
    )
    return selected, rows


def _trajectory_to_camera_and_world(
    trajectory_pose9d: np.ndarray,
    scene_origin: np.ndarray,
    scene_scale: float,
    world_to_camera: np.ndarray,
    object_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_deltas, world_poses = [], []
    for local_matrix in pose9d_to_matrix_np(trajectory_pose9d):
        camera_delta = local_delta_to_camera(local_matrix, scene_origin, scene_scale)
        world_delta = camera_delta_to_world_delta(camera_delta, world_to_camera)
        camera_deltas.append(camera_delta)
        world_poses.append(world_delta @ object_to_world)
    return np.stack(camera_deltas).astype(np.float32), np.stack(world_poses).astype(np.float32)


def _draw_trajectory_axes(
    rgb: np.ndarray,
    manipulated_pixels: np.ndarray,
    reference_pixels: np.ndarray,
    camera_deltas: np.ndarray,
    object_to_camera: np.ndarray,
    scene_origin: np.ndarray,
    intrinsic: np.ndarray,
    axis_length: float,
    manipulated_label: str,
    reference_label: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    layer = canvas.copy()
    for u, v in reference_pixels:
        cv2.circle(layer, (int(u), int(v)), 2, (255, 0, 255), -1, cv2.LINE_AA)
    for u, v in manipulated_pixels:
        cv2.circle(layer, (int(u), int(v)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.72, layer, 0.28, 0.0)

    object_from_camera = np.linalg.inv(object_to_camera)
    centroid_object = (
        object_from_camera
        @ np.concatenate((scene_origin.astype(np.float32), np.ones(1, dtype=np.float32)))
    )[:3]
    base = np.eye(4, dtype=np.float32)
    base[:3, 3] = centroid_object
    base_endpoints = np.repeat(base[None], 4, axis=0)
    base_endpoints[1, :3, 3] += np.asarray([axis_length, 0, 0])
    base_endpoints[2, :3, 3] += np.asarray([0, axis_length, 0])
    base_endpoints[3, :3, 3] += np.asarray([0, 0, axis_length])
    points_object = base_endpoints[:, :3, 3]

    origins = []
    colors = ((0, 0, 255), (0, 210, 0), (255, 80, 20))
    height, width = canvas.shape[:2]
    for index, camera_delta in enumerate(camera_deltas):
        absolute = camera_delta @ object_to_camera
        moved = points_object @ absolute[:3, :3].T + absolute[:3, 3]
        uv = _project(moved, intrinsic)
        origin = tuple(int(value) for value in uv[0])
        origins.append(uv[0])
        if not (0 <= origin[0] < width and 0 <= origin[1] < height):
            continue
        alpha = 0.30 + 0.70 * index / max(len(camera_deltas) - 1, 1)
        for axis, color in enumerate(colors):
            color_faded = tuple(int(channel * alpha) for channel in color)
            endpoint = tuple(int(value) for value in uv[axis + 1])
            cv2.line(canvas, origin, endpoint, color_faded, 1, cv2.LINE_AA)
        cv2.circle(canvas, origin, 2, (240, 240, 240), -1, cv2.LINE_AA)
        if index % 4 == 0 or index == len(camera_deltas) - 1:
            cv2.putText(
                canvas,
                str(index + 1),
                (origin[0] + 3, origin[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if len(origins) > 1:
        cv2.polylines(
            canvas, [np.asarray(origins, dtype=np.int32)], False, (0, 150, 255), 2, cv2.LINE_AA
        )
    cv2.rectangle(canvas, (0, 0), (width, 66), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        "Stage 2: selected Goal + Full64 trajectory",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{manipulated_label} samples yellow | {reference_label} samples magenta | axes X red / Y green / Z blue",
        (16, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_goal_candidates(
    rgb: np.ndarray,
    goal_pose9d: np.ndarray,
    selected_index: int,
    object_to_camera: np.ndarray,
    scene_origin: np.ndarray,
    scene_scale: float,
    intrinsic: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    camera_from_object = np.asarray(object_to_camera, dtype=np.float32)
    object_from_camera = np.linalg.inv(camera_from_object)
    centroid_object = (
        object_from_camera
        @ np.concatenate((scene_origin.astype(np.float32), np.ones(1, dtype=np.float32)))
    )[:3]
    points_object = np.stack(
        (
            centroid_object,
            centroid_object + np.asarray([axis_length, 0, 0], dtype=np.float32),
            centroid_object + np.asarray([0, axis_length, 0], dtype=np.float32),
            centroid_object + np.asarray([0, 0, axis_length], dtype=np.float32),
        )
    )
    origins = []
    selected_uv = None
    selected_axis_uv = None
    for index, local_matrix in enumerate(pose9d_to_matrix_np(goal_pose9d)):
        delta = local_delta_to_camera(local_matrix, scene_origin, scene_scale)
        absolute = delta @ camera_from_object
        moved = points_object @ absolute[:3, :3].T + absolute[:3, 3]
        uv = _project(moved, intrinsic)
        origins.append(uv[0])
        if index == selected_index:
            selected_uv = uv[0]
            selected_axis_uv = uv
        else:
            cv2.circle(canvas, tuple(uv[0]), 3, (175, 175, 175), -1, cv2.LINE_AA)
    if origins:
        hull_points = np.asarray(origins, dtype=np.int32)
        if len(hull_points) >= 3:
            cv2.polylines(canvas, [cv2.convexHull(hull_points)], True, (150, 150, 150), 1, cv2.LINE_AA)
    if selected_axis_uv is not None:
        colors = ((0, 0, 255), (0, 210, 0), (255, 80, 20))
        start = tuple(int(v) for v in selected_axis_uv[0])
        for axis, color in enumerate(colors):
            endpoint = tuple(int(v) for v in selected_axis_uv[axis + 1])
            cv2.line(canvas, start, endpoint, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, start, 5, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"selected goal {selected_index}", (start[0] + 7, start[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (18, 18, 18), -1)
    cv2.putText(canvas, "Goal Diffusion candidates on simulator RGB", (16, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "gray: all goals | colored axes: selected terminal pose", (16, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _heat_overlay(
    rgb: np.ndarray,
    pixels: np.ndarray,
    importance: np.ndarray,
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    values = np.asarray(importance, dtype=np.float32)
    values = (values - values.min()) / max(float(values.max() - values.min()), 1e-8)
    layer = canvas.copy()
    for (u, v), value in zip(pixels, values):
        radius = 2 + int(round(5 * float(value)))
        scaled = tuple(int(channel * (0.25 + 0.75 * value)) for channel in color)
        cv2.circle(layer, (int(u), int(v)), radius, scaled, -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.58, layer, 0.42, 0.0)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (18, 18, 18), -1)
    cv2.putText(canvas, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="pouring")
    parser.add_argument("--manipulated-label", default="manipulated object")
    parser.add_argument("--reference-label", default="reference object")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--dino-weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-goals", type=int, default=16)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--goal-inference-steps", type=int)
    parser.add_argument("--trajectory-inference-steps", type=int)
    parser.add_argument("--axis-length", type=float, default=0.018)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, config, payload = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=not args.no_ema
    )
    cache_root = Path(args.cache_root or config["data"]["cache_root"]).expanduser().resolve()
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    weights = Path(args.dino_weights or manifest["dino"]["weights"]).expanduser().resolve()
    snapshot = np.load(Path(args.snapshot).expanduser().resolve(), allow_pickle=False)
    manipulated_mask_key = _first_existing(snapshot, ("manipulated_mask", "cup_mask"))
    reference_mask_key = _first_existing(snapshot, ("reference_mask", "bowl_mask"))
    object_world_key = _first_existing(snapshot, ("T_manipulated_to_world", "T_object_to_world"))
    object_camera_key = _first_existing(snapshot, ("T_manipulated_to_camera", "T_object_to_camera"))
    rgb = np.asarray(snapshot["rgb"], dtype=np.uint8)
    depth = np.asarray(snapshot["depth_m"], dtype=np.float32)
    intrinsic = np.asarray(snapshot["intrinsic_cv"], dtype=np.float32)

    manipulated_pixels = farthest_pixel_sample(
        valid_mask_pixels(snapshot[manipulated_mask_key], depth), 256
    )
    reference_pixels = farthest_pixel_sample(
        valid_mask_pixels(snapshot[reference_mask_key], depth), 256
    )
    manipulated_camera = unproject_pixels(manipulated_pixels, depth, intrinsic)
    reference_camera = unproject_pixels(reference_pixels, depth, intrinsic)
    scene_origin = manipulated_camera.mean(0).astype(np.float32)
    scene_scale = float(manifest.get("scene_scale", 1.0))
    manipulated_local = ((manipulated_camera - scene_origin) / scene_scale).astype(np.float32)
    reference_local = ((reference_camera - scene_origin) / scene_scale).astype(np.float32)

    extractor = DinoV2DenseExtractor(weights_path=weights, device=str(device))
    grid, padded_shape = extract_dino_grid(extractor, rgb)
    manipulated_dino = sample_dense_features(grid, manipulated_pixels, padded_shape)
    reference_dino = sample_dense_features(grid, reference_pixels, padded_shape)
    assert_unique_aligned(manipulated_pixels, manipulated_local, manipulated_dino, 256)
    assert_unique_aligned(reference_pixels, reference_local, reference_dino, 256)

    batch = {
        "manipulated_points": torch.from_numpy(manipulated_local)[None].to(device),
        "manipulated_dino": torch.from_numpy(manipulated_dino)[None].to(device),
        "reference_points": torch.from_numpy(reference_local)[None].to(device),
        "reference_dino": torch.from_numpy(reference_dino)[None].to(device),
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    samples, encoding = model.sample(
        batch,
        num_goal_samples=args.num_goals,
        num_trajectory_samples=args.num_trajectories,
        generator=generator,
        return_debug=True,
        goal_inference_steps=args.goal_inference_steps,
        trajectory_inference_steps=args.trajectory_inference_steps,
    )
    goals = samples.goals[0].cpu().numpy().astype(np.float32)
    trajectories = samples.trajectories[0].cpu().numpy().astype(np.float32)
    prior = _load_training_prior(cache_root)
    selected, ranking = _rank_candidates(goals, trajectories, reference_local.mean(0), prior)
    goal_index, trajectory_index = (int(selected[0]), int(selected[1]))
    selected_trajectory = trajectories[goal_index, trajectory_index]
    camera_deltas, world_poses = _trajectory_to_camera_and_world(
        selected_trajectory,
        scene_origin,
        scene_scale,
        np.asarray(snapshot["T_world_to_camera"], dtype=np.float32),
        np.asarray(snapshot[object_world_key], dtype=np.float32),
    )
    camera_to_world = np.linalg.inv(
        np.asarray(snapshot["T_world_to_camera"], dtype=np.float32)
    )
    reference_center_camera = reference_camera.mean(0)
    reference_center_world = (
        camera_to_world
        @ np.concatenate(
            (reference_center_camera, np.ones(1, dtype=np.float32))
        )
    )[:3].astype(np.float32)
    final_to_reference = world_poses[-1, :3, 3] - reference_center_world

    overlay = _draw_trajectory_axes(
        rgb,
        manipulated_pixels,
        reference_pixels,
        camera_deltas,
        np.asarray(snapshot[object_camera_key], dtype=np.float32),
        scene_origin,
        intrinsic,
        args.axis_length,
        args.manipulated_label,
        args.reference_label,
    )
    overlay_path = output / "full64_coordinate_frames_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    goal_overlay = _draw_goal_candidates(
        rgb,
        goals,
        goal_index,
        np.asarray(snapshot[object_camera_key], dtype=np.float32),
        scene_origin,
        scene_scale,
        intrinsic,
        args.axis_length * 1.5,
    )
    goal_overlay_path = output / "goal_pose_candidates_overlay.png"
    cv2.imwrite(str(goal_overlay_path), goal_overlay)
    # V2/V6 expose directional relevance as ``*_importance``.  V7 makes the
    # scalar Motion Functional Field the only generator-facing relation, so
    # use that field for the same diagnostic overlay.
    manipulated_relation = encoding.manipulated_importance
    if manipulated_relation is None:
        manipulated_relation = encoding.manipulated_motion_field
    reference_relation = encoding.reference_importance
    if reference_relation is None:
        reference_relation = encoding.reference_motion_field
    if manipulated_relation is None or reference_relation is None:
        raise RuntimeError("Checkpoint does not expose relation fields for visualization")
    manipulated_importance = manipulated_relation[0].cpu().numpy()
    reference_importance = reference_relation[0].cpu().numpy()
    manipulated_attention = _heat_overlay(
        rgb, manipulated_pixels, manipulated_importance, (0, 200, 255),
        f"reference queries -> important {args.manipulated_label} points",
    )
    reference_attention = _heat_overlay(
        rgb, reference_pixels, reference_importance, (255, 0, 255),
        f"manipulated queries -> important {args.reference_label} points",
    )
    attention_summary = np.concatenate((manipulated_attention, reference_attention), axis=1)
    attention_path = output / "encoder_cross_attention_summary.png"
    cv2.imwrite(str(attention_path), attention_summary)
    # Keep a single, fixed artifact for fast qualitative iteration.  The two
    # upper panels show what was sampled/selected and the lower row shows which
    # object regions the shared scene encoder actually used.
    top_row = np.concatenate((goal_overlay, overlay), axis=1)
    if attention_summary.shape[1] != top_row.shape[1]:
        attention_summary = cv2.resize(
            attention_summary,
            (top_row.shape[1], attention_summary.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    inference_summary = np.concatenate((top_row, attention_summary), axis=0)
    summary_path = output / "simulation_inference_summary.png"
    cv2.imwrite(str(summary_path), inference_summary)

    prediction_path = output / "functional_motion_prediction.npz"
    np.savez_compressed(
        prediction_path,
        pred_manipulated_poses_world=world_poses,
        pred_object_poses_world=world_poses,
        pred_camera_deltas=camera_deltas,
        pred_local_poses9d=selected_trajectory,
        selected_goal_pose9d=goals[goal_index],
        all_goal_pose9d=goals,
        all_trajectory_pose9d=trajectories,
        selected_goal_index=np.asarray(goal_index),
        selected_trajectory_index=np.asarray(trajectory_index),
        manipulated_points_camera=manipulated_camera,
        reference_points_camera=reference_camera,
        manipulated_points_local=manipulated_local,
        reference_points_local=reference_local,
        manipulated_pixels_uv=manipulated_pixels,
        reference_pixels_uv=reference_pixels,
        manipulated_dino=manipulated_dino.astype(np.float16),
        reference_dino=reference_dino.astype(np.float16),
        scene_origin=scene_origin,
        scene_scale=np.asarray(scene_scale, dtype=np.float32),
        manipulated_importance=manipulated_importance,
        reference_importance=reference_importance,
    )
    report = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "weights": "raw" if args.no_ema else "ema",
        "snapshot": str(Path(args.snapshot).expanduser().resolve()),
        "seed": args.seed,
        "input_contract": {
            "manipulated_points": [1, 256, 3],
            "manipulated_dino": [1, 256, int(manipulated_dino.shape[-1])],
            "reference_points": [1, 256, 3],
            "reference_dino": [1, 256, int(reference_dino.shape[-1])],
            "same_pixel_indices_for_xyz_and_dino": True,
        },
        "num_goal_samples": args.num_goals,
        "num_trajectories_per_goal": args.num_trajectories,
        "goal_inference_steps": int(
            args.goal_inference_steps or model.goal_diffuser.inference_steps
        ),
        "trajectory_inference_steps": int(
            args.trajectory_inference_steps
            or model.trajectory_diffuser.inference_steps
        ),
        "selected": ranking[0],
        "top_five_candidates": ranking[:5],
        "predicted_final_position_world_m": world_poses[-1, :3, 3].tolist(),
        "visible_reference_center_world_m": reference_center_world.tolist(),
        "predicted_final_to_visible_reference_center_m": float(
            np.linalg.norm(final_to_reference)
        ),
        "predicted_final_to_visible_reference_center_planar_m": float(
            np.linalg.norm(final_to_reference[:2])
        ),
        "predicted_relative_rotation_deg": float(
            np.rad2deg(_rotation_angle(camera_deltas[-1]))
        ),
        "first_step_diagnostics": {
            "local_translation_m": float(
                np.linalg.norm(
                    selected_trajectory[1, :3] - selected_trajectory[0, :3]
                )
            ),
            "local_rotation_deg": float(
                np.rad2deg(
                    _rotation_distance(
                        pose9d_to_matrix_np(selected_trajectory[0]),
                        pose9d_to_matrix_np(selected_trajectory[1]),
                    )
                )
            ),
            "world_translation_m": float(
                np.linalg.norm(world_poses[1, :3, 3] - world_poses[0, :3, 3])
            ),
            "world_z_delta_m": float(
                world_poses[1, 2, 3] - world_poses[0, 2, 3]
            ),
            "training_first_step_p95_m": float(prior["first_step_p95"]),
        },
        "outputs": {
            "prediction": str(prediction_path),
            "trajectory_overlay": str(overlay_path),
            "goal_pose_overlay": str(goal_overlay_path),
            "encoder_attention": str(attention_path),
            "simulation_summary": str(summary_path),
        },
    }
    (output / "motion_inference_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
