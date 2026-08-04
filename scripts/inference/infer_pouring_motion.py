#!/usr/bin/env python3
"""Run the trained goal-pose and Full64 pouring checkpoints on an LFV scene."""

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

from lfv.inference.functional_motion.two_stage_pouring import (
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
        raise ValueError(f"Expected pooled language embedding [1,1024], got {embedding.shape}")
    return embedding


def _load_policy(checkpoint: Path, device: torch.device, *, standalone_normalizer: bool):
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


def _local_pose_for_new_centroid(
    pose_local_old: np.ndarray,
    old_centroid: np.ndarray,
    new_centroid: np.ndarray,
) -> np.ndarray:
    camera_delta = local_pose7d_to_camera_delta(pose_local_old, old_centroid)
    rotation = camera_delta[:3, :3]
    translation_local = (
        rotation @ new_centroid + camera_delta[:3, 3] - new_centroid
    )
    local = camera_delta.copy()
    local[:3, 3] = translation_local
    return matrix_to_pose7d_xyzw(local)


def _project(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float32)
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        (
            points[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2],
            points[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2],
        ),
        axis=-1,
    ).astype(np.int32)


def _draw_input_and_trajectory(
    rgb: np.ndarray,
    manipulated_pixels: np.ndarray,
    target_pixels: np.ndarray,
    local_poses: np.ndarray,
    centroid_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for u, v in target_pixels:
        cv2.circle(canvas, (int(u), int(v)), 2, (255, 0, 255), -1, cv2.LINE_AA)
    for u, v in manipulated_pixels:
        cv2.circle(canvas, (int(u), int(v)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    centers = []
    for pose in local_poses:
        delta = local_pose7d_to_camera_delta(pose, centroid_camera)
        center = delta[:3, :3] @ centroid_camera + delta[:3, 3]
        centers.append(center)
    pixels = _project(np.asarray(centers), intrinsic)
    height, width = canvas.shape[:2]
    for p0, p1 in zip(pixels[:-1], pixels[1:]):
        if (
            -width <= p0[0] <= 2 * width
            and -height <= p0[1] <= 2 * height
            and -width <= p1[0] <= 2 * width
            and -height <= p1[1] <= 2 * height
        ):
            cv2.line(canvas, tuple(p0), tuple(p1), (0, 128, 255), 2, cv2.LINE_AA)
    if len(pixels):
        cv2.circle(canvas, tuple(pixels[0]), 6, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pixels[-1]), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "yellow: contact input | magenta: target", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "orange: predicted Full64 cup path", (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--transfer-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-repo",
        default="/home/users1/ljian/object_centric_diffusion",
    )
    parser.add_argument(
        "--goal-checkpoint",
        default=(
            "/home/users1/ljian/object_centric_diffusion/data/outputs_local_goal_pose/"
            "pouring_seed42/20260428_171300/checkpoints/"
            "epoch=0700-val_sample_goal_pos_err_cm=3.086.ckpt"
        ),
    )
    parser.add_argument(
        "--trajectory-checkpoint",
        default=(
            "/home/users1/ljian/object_centric_diffusion/data/outputs_goal_full64/"
            "pouring_seed42/20260429_210924/checkpoints/epoch=1500.ckpt"
        ),
    )
    parser.add_argument(
        "--language-embedding",
        default="/media/ljian/lj/data_3d/pouring/lang_emb.npy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_repo = Path(args.model_repo).expanduser().resolve()
    if str(model_repo) not in sys.path:
        sys.path.insert(0, str(model_repo))
    os.chdir(model_repo)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = np.load(Path(args.snapshot).expanduser(), allow_pickle=False)
    transfer = np.load(Path(args.transfer_result).expanduser(), allow_pickle=False)
    required = ("rgb", "depth_m", "cup_mask", "bowl_mask", "intrinsic_cv",
                "T_world_to_camera", "T_object_to_world")
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise KeyError(f"Snapshot is missing required fields: {missing}")

    heat_key = "target_heatmap" if "target_heatmap" in transfer else "heatmap"
    heat = np.asarray(transfer[heat_key], dtype=np.float32)
    stage1_man, stage1_man_px, stage1_heat = sample_heat_point_cloud(
        heat, snapshot["cup_mask"], snapshot["depth_m"], snapshot["intrinsic_cv"], 256
    )
    stage1_tgt, stage1_tgt_px = sample_mask_point_cloud(
        snapshot["bowl_mask"], snapshot["depth_m"], snapshot["intrinsic_cv"], 256
    )
    stage1_man_local, stage1_tgt_local, centroid1 = localize_clouds(
        stage1_man, stage1_tgt
    )

    stage2_man = stage1_man[:64].copy()
    stage2_tgt = stage1_tgt[:64].copy()
    stage2_man_local, stage2_tgt_local, centroid2 = localize_clouds(
        stage2_man, stage2_tgt
    )
    language = _normalize_lang_embedding(np.load(args.language_embedding))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    goal_checkpoint = Path(args.goal_checkpoint).expanduser().resolve()
    trajectory_checkpoint = Path(args.trajectory_checkpoint).expanduser().resolve()
    goal_policy, goal_cfg, goal_state = _load_policy(
        goal_checkpoint, device, standalone_normalizer=True
    )
    goal_obs = {
        "pc_manipulated": torch.from_numpy(stage1_man_local).unsqueeze(0).to(device),
        "pc_target": torch.from_numpy(stage1_tgt_local).unsqueeze(0).to(device),
        "agent_pos": torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, device=device),
        "lang_token_embs": torch.from_numpy(language).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        goal_result = goal_policy.sample_goal(goal_obs)
    goal_local_stage1 = goal_result["goal_pose7d"][0].detach().cpu().numpy().astype(np.float32)
    goal_local_stage2 = _local_pose_for_new_centroid(
        goal_local_stage1, centroid1, centroid2
    )

    trajectory_policy, trajectory_cfg, trajectory_state = _load_policy(
        trajectory_checkpoint, device, standalone_normalizer=False
    )
    identity = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    goal_transform = local_pose7d_to_camera_delta(goal_local_stage2, np.zeros(3))
    goal_pose9d = _matrix_to_pose9d(goal_transform)
    trajectory_obs = {
        "pc_manipulated": torch.from_numpy(stage2_man_local).unsqueeze(0).unsqueeze(0).to(device),
        "pc_target": torch.from_numpy(stage2_tgt_local).unsqueeze(0).unsqueeze(0).to(device),
        "agent_pos": torch.from_numpy(identity).unsqueeze(0).unsqueeze(0).to(device),
        "goal_pose9d": torch.from_numpy(goal_pose9d).unsqueeze(0).unsqueeze(0).to(device),
        "goal_delta_pose9d": torch.from_numpy(goal_pose9d).unsqueeze(0).unsqueeze(0).to(device),
        "goal_delta_pose7d": torch.from_numpy(goal_local_stage2).unsqueeze(0).unsqueeze(0).to(device),
        "lang_token_embs": torch.from_numpy(language).unsqueeze(0).to(device),
    }
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        trajectory_result = trajectory_policy.predict_action(trajectory_obs)
    local_poses = trajectory_result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
    local_poses[:, 3:7] /= np.maximum(
        np.linalg.norm(local_poses[:, 3:7], axis=-1, keepdims=True), 1e-8
    )
    world_object_poses = local_trajectory_to_world_object_poses(
        local_poses,
        centroid2,
        snapshot["T_world_to_camera"],
        snapshot["T_object_to_world"],
    )

    overlay = _draw_input_and_trajectory(
        snapshot["rgb"], stage2_man_px := stage1_man_px[:64],
        stage2_tgt_px := stage1_tgt_px[:64], local_poses, centroid2,
        snapshot["intrinsic_cv"],
    )
    cv2.imwrite(str(output_dir / "motion_inference_overlay.png"), overlay)
    np.savez_compressed(
        output_dir / "pouring_motion_prediction.npz",
        goal_pose7d_local_stage1=goal_local_stage1,
        goal_pose7d_local_stage2=goal_local_stage2,
        pred_local_poses=local_poses,
        pred_object_poses_world=world_object_poses,
        manipulated_points_camera_stage1=stage1_man,
        target_points_camera_stage1=stage1_tgt,
        manipulated_pixels_uv_stage1=stage1_man_px,
        target_pixels_uv_stage1=stage1_tgt_px,
        manipulated_heat_stage1=stage1_heat,
        manipulated_points_local_stage2=stage2_man_local,
        target_points_local_stage2=stage2_tgt_local,
        manipulated_centroid_camera_stage1=centroid1,
        manipulated_centroid_camera_stage2=centroid2,
    )
    final_rotation = Rotation.from_matrix(world_object_poses[-1, :3, :3])
    initial_rotation = Rotation.from_matrix(snapshot["T_object_to_world"][:3, :3])
    relative_angle = float((final_rotation * initial_rotation.inv()).magnitude())
    report = {
        "task_model": "pouring",
        "selection_reason": "cup-to-container semantics match this scene; pickNplace checkpoint was banana-to-plate",
        "seed": args.seed,
        "device": str(device),
        "goal_checkpoint": str(goal_checkpoint),
        "goal_checkpoint_state": goal_state,
        "goal_checkpoint_training_num_points": int(goal_cfg.task.dataset.num_pts),
        "trajectory_checkpoint": str(trajectory_checkpoint),
        "trajectory_checkpoint_state": trajectory_state,
        "trajectory_checkpoint_training_num_points": int(trajectory_cfg.task.dataset.num_pts),
        "trajectory_steps": int(len(local_poses)),
        "input_contract": {
            "stage1_manipulated": list(stage1_man_local.shape),
            "stage1_target": list(stage1_tgt_local.shape),
            "stage2_manipulated": list(stage2_man_local.shape),
            "stage2_target": list(stage2_tgt_local.shape),
            "manipulated_source": "top continuous transferred heat with valid depth",
            "target_source": "simulator bowl mask with valid depth",
        },
        "goal_pose7d_local_stage1_xyzw": goal_local_stage1.tolist(),
        "goal_pose7d_local_stage2_xyzw": goal_local_stage2.tolist(),
        "predicted_final_object_position_world_m": world_object_poses[-1, :3, 3].tolist(),
        "predicted_relative_rotation_deg": float(np.degrees(relative_angle)),
        "outputs": {
            "prediction": str(output_dir / "pouring_motion_prediction.npz"),
            "overlay": str(output_dir / "motion_inference_overlay.png"),
        },
    }
    (output_dir / "motion_inference_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
