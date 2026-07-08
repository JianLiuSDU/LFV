if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import datetime

import dill
import hydra
import cv2
import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.dataset.lfv_dataset import (
    load_episode_camera_params,
    matrix_to_pose7d_np,
    pose7d_to_matrix_np,
)


CKPT_PATH = os.environ.get(
    "LFV_FULL64_CKPT",
    "data/outputs/full64/pickNplace_lfv_full64_seed42/latest/checkpoints/latest.ckpt",
)
OUTPUT_DIR = os.environ.get("LFV_FULL64_DATASET_OUTPUT_DIR", None)
NUM_SAMPLES = int(os.environ.get("LFV_NUM_DATASET_SAMPLES", "5"))
USE_EMA = os.environ.get("LFV_USE_EMA", "1") != "0"
SAVE_PLOTS = os.environ.get("LFV_SAVE_TRAJ_PLOTS", "1") != "0"
SAVE_IMAGE_OVERLAYS = os.environ.get("LFV_SAVE_IMAGE_OVERLAYS", "1") != "0"
AXIS_LENGTH = float(os.environ.get("LFV_AXIS_LENGTH", "0.05"))
AXIS_DRAW_EVERY = max(1, int(os.environ.get("LFV_AXIS_DRAW_EVERY", "1")))
INTRINSICS_SOURCE = os.environ.get("LFV_INTRINSICS_SOURCE", "depth_intrinsics_original")


def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "now",
        lambda pattern: datetime.datetime.now().strftime(pattern),
        replace=True,
    )
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def to_device(batch, device):
    return dict_apply(batch, lambda x: x.to(device) if hasattr(x, "to") else x)


def load_episode_first_frame_bgr(ep_path: str):
    rgb_path = os.path.join(ep_path, "rgb")
    if os.path.exists(rgb_path):
        rgb_zarr = zarr.open(rgb_path, mode="r")
        rgb = np.asarray(rgb_zarr[0], dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mp4_path = os.path.join(ep_path, "camera_0.mp4")
    if os.path.exists(mp4_path):
        cap = cv2.VideoCapture(mp4_path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame

    return None


def local_pose_to_camera_matrix(local_pose7d: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
    local_pose7d = np.asarray(local_pose7d, dtype=np.float32)
    quat = local_pose7d[3:7].astype(np.float32)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    R_local = R.from_quat(quat).as_matrix().astype(np.float32)
    t_local = local_pose7d[:3].astype(np.float32)

    T_cam = np.eye(4, dtype=np.float32)
    T_cam[:3, :3] = R_local
    T_cam[:3, 3] = t_local - R_local @ centroid_0 + centroid_0
    return T_cam


def compose_pose(base_pose7d: np.ndarray, rel_pose7d: np.ndarray) -> np.ndarray:
    T_abs = pose7d_to_matrix_np(base_pose7d) @ pose7d_to_matrix_np(rel_pose7d)
    return matrix_to_pose7d_np(T_abs)


def residual_traj_to_camera_matrices(residual_actions: np.ndarray, start_pose7d_local: np.ndarray, centroid_0: np.ndarray):
    camera_mats = []
    local_poses = []
    for rel_pose in residual_actions[:, :7]:
        abs_local_pose = compose_pose(start_pose7d_local, rel_pose)
        if local_poses and np.dot(abs_local_pose[3:7], local_poses[-1][3:7]) < 0:
            abs_local_pose[3:7] *= -1
        local_poses.append(abs_local_pose.astype(np.float32))
        camera_mats.append(local_pose_to_camera_matrix(abs_local_pose, centroid_0))
    return np.stack(camera_mats, axis=0), np.stack(local_poses, axis=0)


def project_points(pts_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = np.maximum(pts_3d[:, 2], 1e-6)
    u = pts_3d[:, 0] * fx / z + cx
    v = pts_3d[:, 1] * fy / z + cy
    return np.stack([u, v], axis=-1).astype(np.int32)


def draw_3d_axes_on_image(
    img_bgr: np.ndarray,
    T_camera: np.ndarray,
    intrinsics: np.ndarray,
    centroid_ref: np.ndarray,
    scale: float = 0.05,
    mode: str = "pred",
):
    axes_3d_local = np.array(
        [[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]],
        dtype=np.float32,
    )
    axes_3d_anchored = axes_3d_local + centroid_ref
    axes_3d_anchored_h = np.concatenate(
        [axes_3d_anchored, np.ones((4, 1), dtype=np.float32)],
        axis=1,
    )
    axes_cam = (T_camera @ axes_3d_anchored_h.T).T[:, :3]
    pts_2d = project_points(axes_cam, intrinsics)
    origin, pt_x, pt_y, pt_z = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]

    if mode == "start":
        color_x = color_y = color_z = (255, 255, 0)
        circle_color = (255, 255, 0)
        thickness = 2
    elif mode == "gt":
        color_x = color_y = color_z = (0, 180, 0)
        circle_color = (0, 180, 0)
        thickness = 1
    elif mode == "pred":
        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)
        circle_color = (255, 255, 255)
        thickness = 1
    else:
        color_x = color_y = color_z = (180, 180, 180)
        circle_color = (180, 180, 180)
        thickness = 1

    cv2.line(img_bgr, tuple(origin), tuple(pt_x), color_x, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, tuple(origin), tuple(pt_y), color_y, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, tuple(origin), tuple(pt_z), color_z, thickness, cv2.LINE_AA)
    cv2.circle(img_bgr, tuple(origin), 2, circle_color, -1)
    return origin


def draw_polyline(img_bgr: np.ndarray, points_2d, color=(0, 255, 255), thickness=2):
    if len(points_2d) < 2:
        return
    h, w = img_bgr.shape[:2]
    for p0, p1 in zip(points_2d[:-1], points_2d[1:]):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        if (
            (-w <= x0 <= 2 * w and -h <= y0 <= 2 * h)
            or (-w <= x1 <= 2 * w and -h <= y1 <= 2 * h)
        ):
            cv2.line(img_bgr, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def save_image_trajectory_overlay(
    ep_path: str,
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    start_pose7d_local: np.ndarray,
    centroid_0: np.ndarray,
    out_prefix: str,
    intrinsics_source: str,
):
    if not SAVE_IMAGE_OVERLAYS:
        return "", ""

    img_bgr = load_episode_first_frame_bgr(ep_path)
    if img_bgr is None:
        print(f"[Warn] missing first RGB frame for overlay: {ep_path}")
        return "", ""

    intrinsics, _depth_scale = load_episode_camera_params(ep_path, intrinsics_source)
    pred_mats, _pred_local = residual_traj_to_camera_matrices(pred_actions, start_pose7d_local, centroid_0)
    gt_mats, _gt_local = residual_traj_to_camera_matrices(gt_actions, start_pose7d_local, centroid_0)

    def render(mats, mode, label, line_color):
        canvas = img_bgr.copy()
        origins = []
        for i, T_cam in enumerate(mats):
            draw_mode = "start" if i == 0 else mode
            origin = draw_3d_axes_on_image(
                canvas,
                T_cam,
                intrinsics,
                centroid_ref=centroid_0,
                scale=AXIS_LENGTH,
                mode=draw_mode if (i % AXIS_DRAW_EVERY == 0 or i == len(mats) - 1) else "silent",
            )
            origins.append(origin)
        draw_polyline(canvas, origins, color=line_color, thickness=2)
        cv2.putText(
            canvas,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    pred_path = f"{out_prefix}_pred_overlay.png"
    gt_path = f"{out_prefix}_gt_overlay.png"
    cv2.imwrite(pred_path, render(pred_mats, "pred", "Pred full64 trajectory", (0, 255, 255)))
    cv2.imwrite(gt_path, render(gt_mats, "gt", "GT full64 trajectory", (0, 255, 255)))
    return pred_path, gt_path


def save_trajectory_plot(pred: np.ndarray, gt: np.ndarray, trans_err_cm: np.ndarray, path: str, title: str):
    if not SAVE_PLOTS:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable; skip trajectory plot: {exc}")
        return None

    pred_xyz = pred[: gt.shape[0], :3]
    gt_xyz = gt[:, :3]
    steps = np.arange(gt_xyz.shape[0])

    fig = plt.figure(figsize=(12, 4.5), dpi=140)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="#1f77b4", linewidth=2.0, label="GT")
    ax3d.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color="#d62728", linewidth=2.0, label="Pred")
    ax3d.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="#2ca02c", s=30, label="Start")
    ax3d.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], color="#1f77b4", s=35, marker="x", label="GT end")
    ax3d.scatter(pred_xyz[-1, 0], pred_xyz[-1, 1], pred_xyz[-1, 2], color="#d62728", s=35, marker="x", label="Pred end")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title(title)
    ax3d.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(steps, gt_xyz[:, 0], color="#1f77b4", linestyle="-", label="GT x")
    ax.plot(steps, pred_xyz[:, 0], color="#1f77b4", linestyle="--", label="Pred x")
    ax.plot(steps, gt_xyz[:, 1], color="#ff7f0e", linestyle="-", label="GT y")
    ax.plot(steps, pred_xyz[:, 1], color="#ff7f0e", linestyle="--", label="Pred y")
    ax.plot(steps, gt_xyz[:, 2], color="#2ca02c", linestyle="-", label="GT z")
    ax.plot(steps, pred_xyz[:, 2], color="#2ca02c", linestyle="--", label="Pred z")
    ax_err = ax.twinx()
    ax_err.plot(steps, trans_err_cm, color="#d62728", alpha=0.35, label="Err cm")
    ax.set_xlabel("step")
    ax.set_ylabel("translation (m)")
    ax_err.set_ylabel("translation error (cm)")
    ax.set_title(f"mean {float(np.mean(trans_err_cm)):.2f} cm | max {float(np.max(trans_err_cm)):.2f} cm")
    lines, labels = ax.get_legend_handles_labels()
    err_lines, err_labels = ax_err.get_legend_handles_labels()
    ax.legend(lines + err_lines, labels + err_labels, loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def load_policy_and_dataset(device):
    register_omegaconf_resolvers()
    payload = torch.load(CKPT_PATH, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)

    policy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload.get("state_dicts", {})
    if USE_EMA and "ema_model" in state_dicts:
        policy.load_state_dict(state_dicts["ema_model"], strict=True)
        print("[*] Loaded ema_model")
    else:
        policy.load_state_dict(state_dicts["model"], strict=True)
        print("[*] Loaded model")

    train_dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = train_dataset.get_validation_dataset()
    policy.set_normalizer(train_dataset.get_normalizer())
    policy.eval().to(device)
    return policy, train_dataset, val_dataset, cfg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, _train_dataset, val_dataset, cfg = load_policy_and_dataset(device)

    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = os.path.join(
            str(pathlib.Path(CKPT_PATH).resolve().parents[1]),
            "full64_val_dataset_inference",
        )
    os.makedirs(output_dir, exist_ok=True)

    count = min(NUM_SAMPLES, len(val_dataset))
    if count <= 0:
        raise RuntimeError("Validation dataset is empty; check configs/model/task/multitask_goal_full64.yaml")

    summary = []
    print(f"[*] checkpoint={CKPT_PATH}")
    print(f"[*] val_dataset={len(val_dataset)}, saving {count} samples to {output_dir}")

    for idx in range(count):
        sample = val_dataset[idx]
        batch = {}
        for key, value in sample.items():
            if isinstance(value, dict):
                batch[key] = {
                    k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v
                    for k, v in value.items()
                }
            elif hasattr(value, "unsqueeze"):
                batch[key] = value.unsqueeze(0)
            else:
                batch[key] = value
        batch = to_device(batch, device)

        with torch.no_grad():
            result = policy.predict_action(batch["obs"], target=batch.get("action", None))

        pred = result["action_pred"][0].detach().cpu().numpy().astype(np.float32)
        pred_exec = result["action"][0].detach().cpu().numpy().astype(np.float32)
        gt = batch["action"][0].detach().cpu().numpy().astype(np.float32)

        trans_err_cm = np.linalg.norm(pred[: gt.shape[0], :3] - gt[:, :3], axis=-1) * 100.0
        row = {
            "idx": int(idx),
            "traj_idx": int(sample.get("traj_idx", idx)),
            "mean_trans_err_cm": float(np.mean(trans_err_cm)),
            "max_trans_err_cm": float(np.max(trans_err_cm)),
        }
        ep_path = ""
        if hasattr(val_dataset, "episode_paths") and idx < len(val_dataset.episode_paths):
            ep_path = val_dataset.episode_paths[idx]

        plot_path = os.path.join(output_dir, f"sample_{idx:03d}_traj.png")
        saved_plot_path = save_trajectory_plot(
            pred=pred,
            gt=gt,
            trans_err_cm=trans_err_cm,
            path=plot_path,
            title=f"sample {idx:03d} | traj_idx {row['traj_idx']}",
        )
        row["plot_path"] = saved_plot_path or ""
        row["pred_overlay_path"] = ""
        row["gt_overlay_path"] = ""

        if ep_path:
            start_pose7d_local = sample["obs"]["agent_pos"][0].detach().cpu().numpy().astype(np.float32)
            centroid_0 = sample.get("centroid_0", torch.zeros(3)).detach().cpu().numpy().astype(np.float32)
            overlay_prefix = os.path.join(output_dir, f"sample_{idx:03d}")
            pred_overlay_path, gt_overlay_path = save_image_trajectory_overlay(
                ep_path=ep_path,
                pred_actions=pred,
                gt_actions=gt,
                start_pose7d_local=start_pose7d_local,
                centroid_0=centroid_0,
                out_prefix=overlay_prefix,
                intrinsics_source=getattr(val_dataset, "intrinsics_source", INTRINSICS_SOURCE),
            )
            row["pred_overlay_path"] = pred_overlay_path
            row["gt_overlay_path"] = gt_overlay_path
        summary.append(row)

        np.savez(
            os.path.join(output_dir, f"sample_{idx:03d}.npz"),
            pred_action_full=pred,
            pred_action_exec=pred_exec,
            gt_action=gt,
            centroid_0=sample.get("centroid_0", torch.zeros(3)).detach().cpu().numpy()
            if hasattr(sample.get("centroid_0", None), "detach")
            else np.zeros(3, dtype=np.float32),
            traj_idx=row["traj_idx"],
            mean_trans_err_cm=row["mean_trans_err_cm"],
            max_trans_err_cm=row["max_trans_err_cm"],
            episode_path=ep_path,
            pred_overlay_path=row["pred_overlay_path"],
            gt_overlay_path=row["gt_overlay_path"],
        )
        print(
            f"[*] sample {idx:03d}: traj_idx={row['traj_idx']} "
            f"mean={row['mean_trans_err_cm']:.2f}cm max={row['max_trans_err_cm']:.2f}cm "
            f"plot={row['plot_path'] or 'none'} "
            f"pred_overlay={row['pred_overlay_path'] or 'none'}"
        )

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("idx,traj_idx,mean_trans_err_cm,max_trans_err_cm,plot_path,pred_overlay_path,gt_overlay_path\n")
        for row in summary:
            f.write(
                f"{row['idx']},{row['traj_idx']},"
                f"{row['mean_trans_err_cm']:.6f},{row['max_trans_err_cm']:.6f},"
                f"{row['plot_path']},{row['pred_overlay_path']},{row['gt_overlay_path']}\n"
            )
    print(f"[*] summary saved: {summary_path}")


if __name__ == "__main__":
    main()
