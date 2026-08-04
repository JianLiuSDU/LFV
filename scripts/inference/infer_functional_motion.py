#!/usr/bin/env python3
"""Run task-neutral GoalPose + Full64 inference on an LFV simulator snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import dill
import hydra
import numpy as np
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.inference.functional_motion.two_stage import (
    local_pose7d_to_camera_delta,
    local_trajectory_to_world_object_poses,
    localize_clouds,
    matrix_to_pose7d_xyzw,
    sample_heat_point_cloud,
    sample_mask_point_cloud,
)


def _matrix_to_pose9d(transform: np.ndarray) -> np.ndarray:
    pose = np.zeros(9, dtype=np.float32)
    pose[:3] = transform[:3, 3]
    pose[3:9] = transform[:3, :3][:, :2].T.reshape(6)
    return pose


def _normalize_lang_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim == 1:
        embedding = embedding[None]
    elif embedding.ndim > 2:
        embedding = embedding.reshape(-1, embedding.shape[-1])
    if embedding.shape[0] != 1:
        embedding = embedding.mean(axis=0, keepdims=True)
    if embedding.shape != (1, 1024):
        raise ValueError(f"Expected language embedding [1,1024], got {embedding.shape}")
    return embedding


def _load_policy(checkpoint: Path, device: torch.device, standalone_normalizer: bool):
    payload = torch.load(str(checkpoint), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    states = payload["state_dicts"]
    state_name = "ema_model" if "ema_model" in states else "model"
    policy.load_state_dict(states[state_name], strict=True)
    if standalone_normalizer:
        policy.normalizer.load_state_dict(payload["normalizer"])
    policy.eval().to(device)
    return policy, cfg, state_name


def _local_pose_for_new_centroid(pose: np.ndarray, old: np.ndarray, new: np.ndarray):
    delta = local_pose7d_to_camera_delta(pose, old)
    local = delta.copy()
    local[:3, 3] = delta[:3, :3] @ new + delta[:3, 3] - new
    return matrix_to_pose7d_xyzw(local)


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        (points[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2],
         points[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2]),
        axis=-1,
    ).astype(np.int32)


def _draw_full64_axes(
    rgb: np.ndarray,
    manipulated_pixels: np.ndarray,
    target_pixels: np.ndarray,
    local_poses: np.ndarray,
    centroid: np.ndarray,
    intrinsic: np.ndarray,
    axis_length: float,
    task: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    layer = canvas.copy()
    for u, v in target_pixels:
        cv2.circle(layer, (int(u), int(v)), 2, (255, 0, 255), -1, cv2.LINE_AA)
    for u, v in manipulated_pixels:
        cv2.circle(layer, (int(u), int(v)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.65, layer, 0.35, 0.0)

    base_axes = np.concatenate(
        [centroid[None], centroid[None] + np.eye(3, dtype=np.float32) * axis_length],
        axis=0,
    )
    origins = []
    axis_colors = ((0, 0, 255), (0, 210, 0), (255, 80, 20))  # x/y/z
    height, width = canvas.shape[:2]
    for index, pose in enumerate(local_poses):
        delta = local_pose7d_to_camera_delta(pose, centroid)
        moved = base_axes @ delta[:3, :3].T + delta[:3, 3]
        uv = _project(moved, intrinsic)
        origin = tuple(uv[0])
        origins.append(uv[0])
        if not (0 <= origin[0] < width and 0 <= origin[1] < height):
            continue
        for axis, color in enumerate(axis_colors):
            endpoint = tuple(uv[axis + 1])
            cv2.line(canvas, origin, endpoint, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, origin, 2, (240, 240, 240), -1, cv2.LINE_AA)
        if index % 4 == 0 or index == len(local_poses) - 1:
            cv2.putText(canvas, str(index + 1), (origin[0] + 2, origin[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
    if len(origins) > 1:
        cv2.polylines(canvas, [np.asarray(origins, np.int32)], False, (0, 145, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width, 62), (18, 18, 18), -1)
    cv2.putText(canvas, f"{task}: predicted Full64 manipulated-part trajectory",
                (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "axes: X red | Y green | Z blue; labels every 4 steps",
                (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _first_existing(data, keys):
    for key in keys:
        if key in data:
            return key
    raise KeyError(f"Snapshot contains none of {keys}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--transfer-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-repo", default="/home/users1/ljian/object_centric_diffusion")
    parser.add_argument("--goal-checkpoint", required=True)
    parser.add_argument("--trajectory-checkpoint", required=True)
    parser.add_argument("--language-embedding", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--axis-length", type=float, default=0.012)
    args = parser.parse_args()

    model_repo = Path(args.model_repo).expanduser().resolve()
    sys.path.insert(0, str(model_repo))
    os.chdir(model_repo)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = np.load(Path(args.snapshot).expanduser(), allow_pickle=False)
    transfer = np.load(Path(args.transfer_result).expanduser(), allow_pickle=False)
    man_mask_key = _first_existing(snapshot, ("manipulated_mask", "cup_mask"))
    ref_mask_key = _first_existing(snapshot, ("reference_mask", "bowl_mask"))
    initial_pose_key = _first_existing(snapshot, ("T_manipulated_to_world", "T_object_to_world"))
    heat_key = "target_heatmap" if "target_heatmap" in transfer else "heatmap"

    heat = np.asarray(transfer[heat_key], dtype=np.float32)
    man1, man_px1, man_heat1 = sample_heat_point_cloud(
        heat, snapshot[man_mask_key], snapshot["depth_m"], snapshot["intrinsic_cv"], 256
    )
    ref1, ref_px1 = sample_mask_point_cloud(
        snapshot[ref_mask_key], snapshot["depth_m"], snapshot["intrinsic_cv"], 256
    )
    man_local1, ref_local1, centroid1 = localize_clouds(man1, ref1)
    man2, ref2 = man1[:64].copy(), ref1[:64].copy()
    man_local2, ref_local2, centroid2 = localize_clouds(man2, ref2)
    language = _normalize_lang_embedding(np.load(args.language_embedding))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    goal_ckpt = Path(args.goal_checkpoint).expanduser().resolve()
    traj_ckpt = Path(args.trajectory_checkpoint).expanduser().resolve()
    goal_policy, goal_cfg, goal_state = _load_policy(goal_ckpt, device, True)
    goal_obs = {
        "pc_manipulated": torch.from_numpy(man_local1).unsqueeze(0).to(device),
        "pc_target": torch.from_numpy(ref_local1).unsqueeze(0).to(device),
        "agent_pos": torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, device=device),
        "lang_token_embs": torch.from_numpy(language).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        goal_result = goal_policy.sample_goal(goal_obs)
    goal_local1 = goal_result["goal_pose7d"][0].detach().cpu().numpy().astype(np.float32)
    goal_local2 = _local_pose_for_new_centroid(goal_local1, centroid1, centroid2)

    traj_policy, traj_cfg, traj_state = _load_policy(traj_ckpt, device, False)
    identity = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    goal_transform = local_pose7d_to_camera_delta(goal_local2, np.zeros(3))
    goal_pose9d = _matrix_to_pose9d(goal_transform)
    traj_obs = {
        "pc_manipulated": torch.from_numpy(man_local2).unsqueeze(0).unsqueeze(0).to(device),
        "pc_target": torch.from_numpy(ref_local2).unsqueeze(0).unsqueeze(0).to(device),
        "agent_pos": torch.from_numpy(identity).unsqueeze(0).unsqueeze(0).to(device),
        "goal_pose9d": torch.from_numpy(goal_pose9d).unsqueeze(0).unsqueeze(0).to(device),
        "goal_delta_pose9d": torch.from_numpy(goal_pose9d).unsqueeze(0).unsqueeze(0).to(device),
        "goal_delta_pose7d": torch.from_numpy(goal_local2).unsqueeze(0).unsqueeze(0).to(device),
        "lang_token_embs": torch.from_numpy(language).unsqueeze(0).to(device),
    }
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        trajectory_result = traj_policy.predict_action(traj_obs)
    local_poses = trajectory_result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
    local_poses[:, 3:7] /= np.maximum(np.linalg.norm(local_poses[:, 3:7], axis=-1, keepdims=True), 1e-8)
    world_poses = local_trajectory_to_world_object_poses(
        local_poses, centroid2, snapshot["T_world_to_camera"], snapshot[initial_pose_key]
    )

    overlay = _draw_full64_axes(
        snapshot["rgb"], man_px1[:64], ref_px1[:64], local_poses, centroid2,
        snapshot["intrinsic_cv"], args.axis_length, args.task,
    )
    overlay_path = output_dir / "full64_coordinate_frames_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    prediction_path = output_dir / "functional_motion_prediction.npz"
    np.savez_compressed(
        prediction_path,
        task=np.asarray(args.task),
        goal_pose7d_local_stage1=goal_local1,
        goal_pose7d_local_stage2=goal_local2,
        pred_local_poses=local_poses,
        pred_manipulated_poses_world=world_poses,
        pred_object_poses_world=world_poses,
        manipulated_points_camera_stage1=man1,
        reference_points_camera_stage1=ref1,
        manipulated_pixels_uv_stage1=man_px1,
        reference_pixels_uv_stage1=ref_px1,
        manipulated_heat_stage1=man_heat1,
        manipulated_points_local_stage2=man_local2,
        reference_points_local_stage2=ref_local2,
        manipulated_centroid_camera_stage1=centroid1,
        manipulated_centroid_camera_stage2=centroid2,
    )
    initial_rot = Rotation.from_matrix(snapshot[initial_pose_key][:3, :3])
    final_rot = Rotation.from_matrix(world_poses[-1, :3, :3])
    report = {
        "task": args.task,
        "seed": args.seed,
        "goal_checkpoint": str(goal_ckpt),
        "goal_checkpoint_state": goal_state,
        "goal_training_num_points": int(goal_cfg.task.dataset.num_pts),
        "trajectory_checkpoint": str(traj_ckpt),
        "trajectory_checkpoint_state": traj_state,
        "trajectory_training_num_points": int(traj_cfg.task.dataset.num_pts),
        "trajectory_steps": int(len(local_poses)),
        "predicted_final_position_world_m": world_poses[-1, :3, 3].tolist(),
        "predicted_relative_rotation_deg": float(np.degrees((final_rot * initial_rot.inv()).magnitude())),
        "outputs": {"prediction": str(prediction_path), "overlay": str(overlay_path)},
    }
    (output_dir / "motion_inference_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
