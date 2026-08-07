#!/usr/bin/env python3
"""Visualize Stage 2 top-1 inference beside GT on cached train episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

from lfv.geometry import (
    local_delta_to_camera,
    pose9d_to_matrix_np,
    rotation_6d_to_matrix,
    so3_geodesic_distance,
)
from lfv.models.functional_motion_generation import load_stage2_checkpoint
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        (
            points[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2],
            points[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2],
        ),
        axis=-1,
    ).astype(np.int32)


def _trajectory_camera_deltas(
    poses: np.ndarray, origin: np.ndarray, scale: float
) -> np.ndarray:
    return np.stack(
        [
            local_delta_to_camera(matrix, origin, scale)
            for matrix in pose9d_to_matrix_np(poses)
        ]
    ).astype(np.float32)


def _draw_points(
    rgb: np.ndarray, manipulated_pixels: np.ndarray, reference_pixels: np.ndarray
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    layer = canvas.copy()
    for u, v in reference_pixels:
        cv2.circle(layer, (int(u), int(v)), 2, (255, 0, 255), -1, cv2.LINE_AA)
    for u, v in manipulated_pixels:
        cv2.circle(layer, (int(u), int(v)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    return cv2.addWeighted(canvas, 0.72, layer, 0.28, 0.0)


def _draw_axes(
    rgb: np.ndarray,
    deltas: np.ndarray,
    origin_camera: np.ndarray,
    intrinsic: np.ndarray,
    *,
    title: str,
    path_color: tuple[int, int, int],
    axis_length: float,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    axes = np.concatenate(
        (
            origin_camera[None],
            origin_camera[None]
            + np.eye(3, dtype=np.float32) * float(axis_length),
        ),
        axis=0,
    )
    origins = []
    colors = ((0, 0, 255), (0, 210, 0), (255, 80, 20))
    height, width = canvas.shape[:2]
    for index, delta in enumerate(deltas):
        moved = axes @ delta[:3, :3].T + delta[:3, 3]
        uv = _project(moved, intrinsic)
        origins.append(uv[0])
        start = tuple(int(v) for v in uv[0])
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            continue
        alpha = 0.28 + 0.72 * index / max(len(deltas) - 1, 1)
        for axis, color in enumerate(colors):
            endpoint = tuple(int(v) for v in uv[axis + 1])
            faded = tuple(int(channel * alpha) for channel in color)
            cv2.line(canvas, start, endpoint, faded, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 2, (245, 245, 245), -1, cv2.LINE_AA)
        if index % 8 == 0 or index == len(deltas) - 1:
            cv2.putText(
                canvas,
                str(index),
                (start[0] + 2, start[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if origins:
        cv2.polylines(
            canvas,
            [np.asarray(origins, dtype=np.int32)],
            False,
            path_color,
            2,
            cv2.LINE_AA,
        )
        if len(origins) > 1:
            cv2.line(
                canvas,
                tuple(origins[0]),
                tuple(origins[1]),
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                tuple(origins[0]),
                tuple(origins[1]),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.rectangle(canvas, (0, 0), (width, 42), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _pose_errors(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = torch.from_numpy(predicted).float()
    gt = torch.from_numpy(target).float()
    translation = torch.linalg.norm(pred[..., :3] - gt[..., :3], dim=-1)
    rotation = so3_geodesic_distance(
        rotation_6d_to_matrix(pred[..., 3:9]),
        rotation_6d_to_matrix(gt[..., 3:9]),
    )
    return {
        "mean_translation_m": float(translation.mean()),
        "endpoint_translation_m": float(translation[-1]),
        "mean_rotation_deg": float(torch.rad2deg(rotation).mean()),
        "endpoint_rotation_deg": float(torch.rad2deg(rotation[-1])),
    }


def _adjacent_rotation_deg(poses: np.ndarray, first: int, second: int) -> float:
    matrices = pose9d_to_matrix_np(poses)
    relative = matrices[first, :3, :3].T @ matrices[second, :3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _panel_header(
    panel: np.ndarray, title: str, subtitle: str, width: int = 640
) -> np.ndarray:
    canvas = panel.copy()
    cv2.rectangle(canvas, (0, 0), (width, 66), (18, 18, 18), -1)
    cv2.putText(canvas, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (14, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (210, 230, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--source-root", default="/media/ljian/lj/data_3d/pouring_lfv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--episodes", nargs="+", default=["episode_152", "episode_90", "episode_33", "episode_12"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-goals", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--axis-length", type=float, default=0.025)
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    split = json.loads((cache_root / "split_manifest.json").read_text(encoding="utf-8"))
    train_ids = set(split["episodes"]["train"])
    unknown = sorted(set(args.episodes) - train_ids)
    if unknown:
        raise ValueError(f"Requested episodes are not in train split: {unknown}")

    device = torch.device(args.device)
    model, _, payload = load_stage2_checkpoint(
        args.checkpoint, device=device, use_ema=args.use_ema
    )
    rows, goal_rows, report_rows = [], [], []
    for row_index, episode_id in enumerate(args.episodes):
        with np.load(cache_root / "episodes" / f"{episode_id}.npz") as data:
            manipulated_points = data["manipulated_points"].astype(np.float32)
            manipulated_dino = data["manipulated_dino"].astype(np.float32)
            reference_points = data["reference_points"].astype(np.float32)
            reference_dino = data["reference_dino"].astype(np.float32)
            gt = data["trajectory_pose9d"].astype(np.float32)
            origin = data["scene_origin"].astype(np.float32)
            scale = float(data["scene_scale"])
            intrinsic = data["intrinsic"].astype(np.float32)
            manipulated_pixels = data["manipulated_pixels_uv"].astype(np.int32)
            reference_pixels = data["reference_pixels_uv"].astype(np.int32)
        rgb = np.asarray(zarr.open(str(source_root / episode_id / "rgb"), mode="r")[0])
        batch = {
            "manipulated_points": torch.from_numpy(manipulated_points)[None].to(device),
            "manipulated_dino": torch.from_numpy(manipulated_dino)[None].to(device),
            "reference_points": torch.from_numpy(reference_points)[None].to(device),
            "reference_dino": torch.from_numpy(reference_dino)[None].to(device),
        }
        generator = torch.Generator(device=device).manual_seed(args.seed + row_index)
        samples, _ = model.sample(
            batch,
            num_goal_samples=args.num_goals,
            num_trajectory_samples=args.num_trajectories,
            generator=generator,
        )
        all_predicted = samples.trajectories[0].cpu().numpy().astype(np.float32)
        predicted_goals = samples.goals[0].cpu().numpy().astype(np.float32)
        top1 = all_predicted[0, 0]
        errors = []
        for goal_index in range(all_predicted.shape[0]):
            for trajectory_index in range(all_predicted.shape[1]):
                metric = _pose_errors(all_predicted[goal_index, trajectory_index], gt)
                score = metric["mean_translation_m"] + np.deg2rad(metric["mean_rotation_deg"]) * 0.05
                errors.append((score, goal_index, trajectory_index, metric))
        errors.sort(key=lambda item: item[0])
        _, best_goal, best_trajectory, best_metrics = errors[0]
        goal_errors = []
        for goal_index, predicted_goal in enumerate(predicted_goals):
            metric = _pose_errors(predicted_goal[None], gt[-1:])
            score = metric["mean_translation_m"] + np.deg2rad(metric["mean_rotation_deg"]) * 0.05
            goal_errors.append((score, goal_index, metric))
        goal_errors.sort(key=lambda item: item[0])
        _, best_goal_pose_index, best_goal_metrics = goal_errors[0]
        top1_goal_metrics = _pose_errors(predicted_goals[:1], gt[-1:])

        gt_delta = _trajectory_camera_deltas(gt, origin, scale)
        pred_delta = _trajectory_camera_deltas(top1, origin, scale)
        gt_first = float(np.linalg.norm(gt[1, :3] - gt[0, :3]))
        pred_first = float(np.linalg.norm(top1[1, :3] - top1[0, :3]))
        gt_first_rotation = _adjacent_rotation_deg(gt, 0, 1)
        pred_first_rotation = _adjacent_rotation_deg(top1, 0, 1)
        top1_metrics = _pose_errors(top1, gt)
        input_panel = _panel_header(
            _draw_points(rgb, manipulated_pixels, reference_pixels),
            f"{episode_id} | train input",
            "yellow manipulated | magenta reference",
        )
        gt_panel = _draw_axes(
            rgb, gt_delta, origin, intrinsic,
            title=(
                f"GT cumulative T(0->k) | first {gt_first * 1000:.2f} mm, "
                f"{gt_first_rotation:.2f} deg"
            ),
            path_color=(60, 220, 60), axis_length=args.axis_length,
        )
        pred_panel = _draw_axes(
            rgb, pred_delta, origin, intrinsic,
            title=(
                f"top-1 inference | first {pred_first * 1000:.1f} mm, "
                f"{pred_first_rotation:.1f} deg"
            ),
            path_color=(0, 150, 255), axis_length=args.axis_length,
        )
        subtitle = (
            f"mean t {top1_metrics['mean_translation_m'] * 100:.2f} cm | "
            f"mean R {top1_metrics['mean_rotation_deg']:.1f} deg | "
            f"best K=({best_goal},{best_trajectory}) {best_metrics['mean_translation_m'] * 100:.2f} cm"
        )
        pred_panel = _panel_header(
            pred_panel,
            (
                f"top-1 inference | first {pred_first * 1000:.1f} mm, "
                f"{pred_first_rotation:.1f} deg"
            ),
            subtitle,
        )
        row = np.concatenate((input_panel, gt_panel, pred_panel), axis=1)
        rows.append(row)
        cv2.imwrite(str(output / f"{episode_id}_gt_vs_top1.png"), row)

        gt_goal_panel = _draw_axes(
            rgb,
            _trajectory_camera_deltas(gt[-1:], origin, scale),
            origin,
            intrinsic,
            title="GT terminal Goal T(0->63)",
            path_color=(60, 220, 60),
            axis_length=args.axis_length * 1.5,
        )
        predicted_goal_panel = _draw_axes(
            rgb,
            _trajectory_camera_deltas(predicted_goals[:1], origin, scale),
            origin,
            intrinsic,
            title="Goal Diffusion top-1 prediction",
            path_color=(0, 150, 255),
            axis_length=args.axis_length * 1.5,
        )
        predicted_goal_panel = _panel_header(
            predicted_goal_panel,
            "Goal Diffusion top-1 prediction",
            (
                f"t {top1_goal_metrics['mean_translation_m'] * 100:.2f} cm | "
                f"R {top1_goal_metrics['mean_rotation_deg']:.1f} deg | "
                f"best-of-{args.num_goals} goal {best_goal_pose_index}"
            ),
        )
        goal_row = np.concatenate((input_panel, gt_goal_panel, predicted_goal_panel), axis=1)
        goal_rows.append(goal_row)
        cv2.imwrite(str(output / f"{episode_id}_goal_gt_vs_top1.png"), goal_row)
        report_rows.append(
            {
                "episode_id": episode_id,
                "gt_first_step_translation_m": gt_first,
                "gt_first_step_rotation_deg": gt_first_rotation,
                "predicted_first_step_translation_m": pred_first,
                "predicted_first_step_rotation_deg": pred_first_rotation,
                "top1": top1_metrics,
                "oracle_best_indices": [best_goal, best_trajectory],
                "oracle_best": best_metrics,
                "goal_top1": top1_goal_metrics,
                "goal_oracle_best_index": best_goal_pose_index,
                "goal_oracle_best": best_goal_metrics,
            }
        )

    summary = np.concatenate(rows, axis=0)
    summary_path = output / "train_inference_gt_vs_top1_summary.png"
    cv2.imwrite(str(summary_path), summary)
    goal_summary = np.concatenate(goal_rows, axis=0)
    goal_summary_path = output / "train_goal_pose_gt_vs_top1_summary.png"
    cv2.imwrite(str(goal_summary_path), goal_summary)
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "weights": "ema" if args.use_ema else "raw",
        "split": "train",
        "pose_semantics": "each frame is a cumulative SE(3) transform T_camera_0_to_k, not an adjacent-frame residual",
        "episodes": report_rows,
        "outputs": {
            "trajectory_summary": str(summary_path),
            "goal_pose_summary": str(goal_summary_path),
        },
    }
    report_path = output / "train_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
