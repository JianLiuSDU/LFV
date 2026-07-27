from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp

from lfv.data_processing.episode_io import iter_processed_episodes


def compute_weighted_rigid_transform_se3(P, Q, weights):
    w_sum = np.sum(weights)
    if w_sum < 1e-6:
        return np.eye(4)

    centroid_P = np.sum(P * weights[:, None], axis=0) / w_sum
    centroid_Q = np.sum(Q * weights[:, None], axis=0) / w_sum
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ np.diag(weights) @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = centroid_Q - R_mat @ centroid_P
    return T


def project_points(pts_3d, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = np.maximum(pts_3d[:, 2], 1e-5)
    x2d = (pts_3d[:, 0] * fx / z) + cx
    y2d = (pts_3d[:, 1] * fy / z) + cy
    return np.stack([x2d, y2d], axis=-1).astype(int)


def draw_3d_axes_on_image(img, T_matrix, intrinsics, centroid_ref, scale=0.05):
    axes_3d_local = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]])
    axes_3d_anchored = axes_3d_local + centroid_ref
    axes_3d_anchored_h = np.concatenate([axes_3d_anchored, np.ones((4, 1))], axis=1)
    axes_cam_t = (T_matrix @ axes_3d_anchored_h.T).T[:, :3]
    pts_2d = project_points(axes_cam_t, intrinsics)
    origin, pt_x, pt_y, pt_z = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]
    cv2.line(img, tuple(origin), tuple(pt_x), (0, 0, 255), 2)
    cv2.line(img, tuple(origin), tuple(pt_y), (0, 255, 0), 2)
    cv2.line(img, tuple(origin), tuple(pt_z), (255, 0, 0), 2)
    cv2.circle(img, tuple(origin), 3, (255, 255, 255), -1)
    return origin


def dense_se3_from_tracking(coords, visibs, cfg):
    T_frames = coords.shape[0]
    P_ref = coords[0]
    T_matrices = np.zeros((T_frames, 4, 4))
    T_matrices[0] = np.eye(4)
    visib_threshold = float(cfg.se3.visib_threshold)
    outlier_std = float(cfg.se3.outlier_std_multiplier)

    for t in range(1, T_frames):
        P_t, W_t = coords[t], visibs[t]
        valid_mask = W_t > visib_threshold
        if np.sum(valid_mask) < 3:
            T_matrices[t] = T_matrices[t - 1]
            continue

        P_ref_v, P_t_v, W_t_v = P_ref[valid_mask], P_t[valid_mask], W_t[valid_mask]
        T_init = compute_weighted_rigid_transform_se3(P_ref_v, P_t_v, W_t_v)
        P_ref_v_h = np.concatenate([P_ref_v, np.ones((len(P_ref_v), 1))], axis=1)
        P_t_hat = (T_init @ P_ref_v_h.T).T[:, :3]
        errors = np.linalg.norm(P_t_v - P_t_hat, axis=1)
        thresh = np.mean(errors) + outlier_std * np.std(errors)
        inlier_mask = errors < thresh

        if np.sum(inlier_mask) >= 3:
            T_final = compute_weighted_rigid_transform_se3(
                P_ref_v[inlier_mask], P_t_v[inlier_mask], np.ones(np.sum(inlier_mask))
            )
        else:
            T_final = T_init
        T_matrices[t] = T_final
    return T_matrices


def resample_for_dp(T_matrices, cfg):
    horizon = int(cfg.se3.dp_horizon)
    pos = T_matrices[:, :3, 3]
    quats = R_scipy.from_matrix(T_matrices[:, :3, :3]).as_quat()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    delta_pos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dots = np.sum(quats[1:] * quats[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    delta_theta = 2 * np.arccos(np.abs(dots))
    S = np.concatenate([[0], np.cumsum(delta_pos + float(cfg.se3.lambda_rot) * delta_theta)])
    S = S + np.linspace(0, 1e-7, len(S))
    S_target = np.linspace(0, S[-1], horizon)

    kind = "cubic" if len(S) >= 4 else "linear"
    pos_interp = interp1d(S, pos, axis=0, kind=kind)(S_target)
    quats_interp = Slerp(S, R_scipy.from_quat(quats))(S_target).as_quat()
    gripper_state = np.zeros((horizon, 1), dtype=np.float32)
    gripper_state[-1, 0] = 1.0
    actions_8d = np.concatenate([pos_interp, quats_interp, gripper_state], axis=1)

    T_resampled = np.zeros((horizon, 4, 4))
    T_resampled[:, 3, 3] = 1.0
    T_resampled[:, :3, 3] = pos_interp
    T_resampled[:, :3, :3] = R_scipy.from_quat(quats_interp).as_matrix()
    return actions_8d, T_resampled


def process_episode(ep_path: str | Path, cfg) -> bool:
    ep_path = Path(ep_path)
    npz_path = ep_path / "point_tracking" / "tapip3d_result.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing tracking file: {npz_path}")

    out_dir = ep_path / "se3_trajectory"
    viz_dir = ep_path / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dp_action_trajectory.npz"
    if out_path.exists() and not bool(cfg.runtime.overwrite):
        print(f"[se3] skip existing {ep_path.name}")
        return True

    data = np.load(npz_path)
    coords = data["coords"]
    visibs = data["visibs"]
    video = data["video"]
    intrinsics = data["intrinsics"]
    centroid_ref = np.mean(coords[0], axis=0)

    T_matrices = dense_se3_from_tracking(coords, visibs, cfg)
    np.savez_compressed(out_dir / "se3_relative_trajectory.npz", T_cam_0_to_t=T_matrices)
    actions_8d, T_resampled = resample_for_dp(T_matrices, cfg)
    np.savez_compressed(out_path, actions_8d=actions_8d, T_matrices_4x4=T_resampled)

    viz_canvas = cv2.cvtColor(video[0], cv2.COLOR_RGB2BGR)
    prev_pt2d = None
    for T_k in T_resampled:
        curr_pt2d = draw_3d_axes_on_image(
            viz_canvas, T_k, intrinsics[0], centroid_ref, scale=float(cfg.se3.axis_length)
        )
        if prev_pt2d is not None:
            cv2.line(viz_canvas, tuple(prev_pt2d), tuple(curr_pt2d), (0, 255, 255), 1)
        prev_pt2d = curr_pt2d
    cv2.imwrite(str(viz_dir / "dp_trajectory_overlay.png"), viz_canvas)
    print(f"[se3] {ep_path.name}: {coords.shape[0]} frames -> {actions_8d.shape[0]} actions")
    return True


def run(cfg) -> None:
    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg)
        except Exception as exc:
            print(f"[se3] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        print(f"[se3] failed {len(failed)} episodes")

