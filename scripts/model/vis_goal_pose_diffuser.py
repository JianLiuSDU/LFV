if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import csv
import glob
import os
import pathlib

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.goal.pose_utils import pose7d_to_matrix, pose9d_to_matrix
from diffusion_policy_3d.policy.goal_pose_diffuser import GoalPoseDiffuser


CKPT_PATH = os.environ.get(
    "LFV_GOAL_CKPT",
    "data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest/checkpoints/latest.ckpt",
)
OUTPUT_DIR = os.environ.get("LFV_GOAL_VIS_DIR", "data/outputs/goal_pose/visualizations")
NUM_SAMPLES = 5
AXIS_LENGTH = 0.05


def pose_to_matrix(pose):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return T


def matrix_to_pose(T):
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = T[:3, 3]
    pose[3:7] = R.from_matrix(T[:3, :3]).as_quat()
    return pose


def local_pose_to_camera_matrix(abs_local_pose, centroid_0):
    T_local = pose_to_matrix(abs_local_pose)
    Rm = T_local[:3, :3]
    t_local = T_local[:3, 3]
    T_cam = T_local.copy()
    T_cam[:3, 3] = t_local - Rm @ centroid_0 + centroid_0
    return T_cam


def project_points(pts_3d, intrinsics):
    z = np.clip(pts_3d[:, 2], 1e-6, None)
    u = intrinsics[0, 0] * pts_3d[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * pts_3d[:, 1] / z + intrinsics[1, 2]
    return np.stack([u, v], axis=1).astype(np.int32)


def draw_3d_axes_on_image(img_bgr, T_matrix, intrinsics, scale=0.05, colors=None):
    if colors is None:
        colors = [(0, 0, 255), (0, 128, 255), (0, 255, 255)]
    pts_local = np.array(
        [[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]],
        dtype=np.float32,
    )
    pts_cam = (T_matrix[:3, :3] @ pts_local.T).T + T_matrix[:3, 3]
    pts_2d = project_points(pts_cam, intrinsics)
    try:
        import cv2

        o = tuple(pts_2d[0])
        for j, color in enumerate(colors):
            cv2.line(img_bgr, o, tuple(pts_2d[j + 1]), color, 2, cv2.LINE_AA)
            cv2.circle(img_bgr, tuple(pts_2d[j + 1]), 3, color, -1)
        return img_bgr
    except Exception:
        return img_bgr


def find_episode_image(dataset, traj_idx):
    if hasattr(dataset, "episode_paths") and traj_idx < len(dataset.episode_paths):
        ep = dataset.episode_paths[traj_idx]
    else:
        candidates = []
        for data_dir in dataset.data_dirs:
            candidates.extend(sorted(glob.glob(os.path.join(data_dir, "episode_*"))))
        if traj_idx >= len(candidates):
            return None
        ep = candidates[traj_idx]
    if ep is None:
        return None
    patterns = [
        os.path.join(ep, "rgb", "0000.png"),
        os.path.join(ep, "rgb", "0.png"),
        os.path.join(ep, "image", "0000.png"),
        os.path.join(ep, "images", "0000.png"),
        os.path.join(ep, "color", "0000.png"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def save_3d_fallback(path, pc_man, pc_tgt, T_pred, T_gt):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc_man[:, 0], pc_man[:, 1], pc_man[:, 2], s=3, c="gray", label="manipulated")
    ax.scatter(pc_tgt[:, 0], pc_tgt[:, 1], pc_tgt[:, 2], s=3, c="blue", label="target")
    for T, color, label in [(T_gt, "green", "gt"), (T_pred, "red", "pred")]:
        origin = T[:3, 3]
        ax.scatter([origin[0]], [origin[1]], [origin[2]], c=color, label=label)
        for k in range(3):
            end = origin + T[:3, k] * AXIS_LENGTH
            ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], c=color)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    payload = torch.load(CKPT_PATH, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()

    model: GoalPoseDiffuser = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload["state_dicts"].get("ema_model", payload["state_dicts"]["model"]), strict=True)
    if "normalizer" in payload:
        model.normalizer.load_state_dict(payload["normalizer"])
    else:
        model.set_normalizer(normalizer)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows = []
    if len(val_dataset) == 0:
        print("Validation dataset is empty; no visualization produced.")
        return

    for i in range(min(NUM_SAMPLES, len(val_dataset))):
        sample = val_dataset[i]
        batch = dict_apply(sample, lambda x: x.unsqueeze(0).to(device) if isinstance(x, torch.Tensor) else x)
        with torch.no_grad():
            result = model.sample_goal(batch["obs"])

        pred9 = result["goal_pose9d"][0].cpu()
        pred7 = result["goal_pose7d"][0].cpu()
        gt9 = sample["goal_pose9d"]
        gt7 = sample["goal_pose7d"]
        T_pred = pose9d_to_matrix(pred9).numpy()
        T_gt = pose9d_to_matrix(gt9).numpy()
        rot_err = GoalPoseDiffuser._rotation_error_per_sample(
            pose9d_to_matrix(pred9.unsqueeze(0))[:, :3, :3],
            pose9d_to_matrix(gt9.unsqueeze(0))[:, :3, :3],
        )[0].item() * 180.0 / np.pi
        pos_err = torch.linalg.norm(pred9[:3] - gt9[:3]).item() * 100.0
        traj_idx = int(sample["traj_idx"].item())
        ep_name = f"episode_{traj_idx:06d}"
        print(f"{ep_name}: pred xyz {pred9[:3].numpy()} | gt xyz {gt9[:3].numpy()} | {pos_err:.2f} cm | {rot_err:.2f} deg")

        out_png = os.path.join(OUTPUT_DIR, f"goal_pose_viz_{i:02d}_episode_{traj_idx}.png")
        image_path = find_episode_image(val_dataset, traj_idx)
        if image_path is not None and val_dataset.intrinsics is not None:
            try:
                import cv2

                img = cv2.imread(image_path)
                centroid_0 = np.asarray(val_dataset.pc_man_0_list[traj_idx]).mean(axis=0)
                T_pred_cam = local_pose_to_camera_matrix(pred7.numpy(), centroid_0)
                T_gt_cam = local_pose_to_camera_matrix(gt7.numpy(), centroid_0)
                draw_3d_axes_on_image(img, T_gt_cam, val_dataset.intrinsics, AXIS_LENGTH, colors=[(0, 255, 0)] * 3)
                draw_3d_axes_on_image(img, T_pred_cam, val_dataset.intrinsics, AXIS_LENGTH, colors=[(0, 0, 255), (0, 128, 255), (0, 165, 255)])
                cv2.imwrite(out_png, img)
            except Exception as exc:
                print(f"Image overlay failed for {ep_name}: {exc}. Saving 3D fallback.")
                save_3d_fallback(out_png, sample["obs"]["pc_manipulated"].numpy(), sample["obs"]["pc_target"].numpy(), T_pred, T_gt)
        else:
            print(f"No RGB image found for {ep_name}; saving 3D fallback/summary only.")
            save_3d_fallback(out_png, sample["obs"]["pc_manipulated"].numpy(), sample["obs"]["pc_target"].numpy(), T_pred, T_gt)

        rows.append(
            {
                "episode": ep_name,
                "pred_xyz": pred9[:3].numpy().tolist(),
                "gt_xyz": gt9[:3].numpy().tolist(),
                "pred_quat": pred7[3:7].numpy().tolist(),
                "gt_quat": gt7[3:7].numpy().tolist(),
                "position_error_cm": pos_err,
                "rotation_error_deg": rot_err,
            }
        )

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
