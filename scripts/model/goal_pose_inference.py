# if __name__ == "__main__":
#     import sys
#     import os
#     import pathlib

#     ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
#     if ROOT_DIR not in sys.path:
#         sys.path.insert(0, ROOT_DIR)
#     os.chdir(ROOT_DIR)

# import os
# import glob
# import csv
# import pathlib

# import cv2
# import dill
# import hydra
# import numpy as np
# import torch
# from omegaconf import OmegaConf
# from scipy.spatial.transform import Rotation as R
# from PIL import Image

# try:
#     import huggingface_hub
#     if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
#         huggingface_hub.cached_download = huggingface_hub.hf_hub_download
# except Exception:
#     pass

# OmegaConf.register_new_resolver("eval", eval, replace=True)

# # ================= 配置区 =================
# CKPT_PATH / OUTPUT_DIR are configured in the active section below.

# NUM_SAMPLES = 10
# AXIS_LENGTH = 0.05
# SEED = 42
# USE_EMA = True
# # ==========================================


# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)


# def resize_points(points: np.ndarray, num_pts: int) -> np.ndarray:
#     """
#     与 GoalPoseSE3Dataset._resize_points 保持一致。
#     注意：centroid_0 必须按训练 dataset 的方式计算，否则 2D overlay 会偏。
#     """
#     points = np.asarray(points, dtype=np.float32)

#     if points.shape[0] == num_pts:
#         return points.astype(np.float32)

#     if points.shape[0] > num_pts:
#         idx = np.linspace(0, points.shape[0] - 1, num_pts).astype(np.int64)
#         return points[idx].astype(np.float32)

#     if points.shape[0] == 0:
#         return np.zeros((num_pts, 3), dtype=np.float32)

#     pad_idx = np.arange(num_pts - points.shape[0]) % points.shape[0]
#     return np.concatenate([points, points[pad_idx]], axis=0).astype(np.float32)


# def pose_to_matrix(pose7d: np.ndarray) -> np.ndarray:
#     """
#     pose7d: [x, y, z, qx, qy, qz, qw]
#     """
#     T = np.eye(4, dtype=np.float32)
#     T[:3, 3] = pose7d[:3]
#     T[:3, :3] = R.from_quat(pose7d[3:7]).as_matrix()
#     return T


# def local_pose_to_camera_matrix(abs_local_pose: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
#     """
#     训练中的 local pose 定义：
#         p_local = p_cam - C0
#         p_local' = R * p_local + t_local

#     转回 camera/global 坐标变换：
#         p_cam' = R * p_cam + t_cam

#     可得：
#         t_cam = t_local - R * C0 + C0
#     """
#     R_local = R.from_quat(abs_local_pose[3:7]).as_matrix()
#     t_local = abs_local_pose[:3]
#     t_cam = t_local - R_local @ centroid_0 + centroid_0

#     T_cam = np.eye(4, dtype=np.float32)
#     T_cam[:3, :3] = R_local
#     T_cam[:3, 3] = t_cam
#     return T_cam


# def project_points(pts_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]

#     z = np.maximum(pts_3d[:, 2], 1e-5)
#     x2d = pts_3d[:, 0] * fx / z + cx
#     y2d = pts_3d[:, 1] * fy / z + cy

#     return np.stack([x2d, y2d], axis=-1).astype(int)


# def draw_3d_axes_on_image(
#     img_bgr: np.ndarray,
#     T_matrix: np.ndarray,
#     intrinsics: np.ndarray,
#     centroid_ref: np.ndarray,
#     scale: float = 0.05,
#     is_gt: bool = False,
# ):
#     """
#     和你原来的轨迹可视化保持一致：
#     先把坐标轴锚定在第一帧 manipulated centroid 上，再经过 T_matrix 变换。
#     这样画出来的是 object centroid 处的坐标系，而不是 camera origin 处的坐标系。
#     """
#     axes_3d_local = np.array(
#         [
#             [0, 0, 0],
#             [scale, 0, 0],
#             [0, scale, 0],
#             [0, 0, scale],
#         ],
#         dtype=np.float32,
#     )

#     axes_3d_anchored = axes_3d_local + centroid_ref
#     axes_3d_anchored_h = np.concatenate(
#         [axes_3d_anchored, np.ones((4, 1), dtype=np.float32)],
#         axis=1,
#     )

#     axes_cam_t = (T_matrix @ axes_3d_anchored_h.T).T[:, :3]
#     pts_2d = project_points(axes_cam_t, intrinsics)

#     origin, pt_x, pt_y, pt_z = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]

#     if is_gt:
#         # GT：绿色细线
#         cv2.line(img_bgr, tuple(origin), tuple(pt_x), (0, 150, 0), 1, cv2.LINE_AA)
#         cv2.line(img_bgr, tuple(origin), tuple(pt_y), (0, 150, 0), 1, cv2.LINE_AA)
#         cv2.line(img_bgr, tuple(origin), tuple(pt_z), (0, 150, 0), 1, cv2.LINE_AA)
#         cv2.circle(img_bgr, tuple(origin), 3, (0, 220, 0), -1)
#     else:
#         # Pred：RGB 三轴
#         cv2.line(img_bgr, tuple(origin), tuple(pt_x), (0, 0, 255), 2, cv2.LINE_AA)      # x red
#         cv2.line(img_bgr, tuple(origin), tuple(pt_y), (0, 255, 0), 2, cv2.LINE_AA)      # y green
#         cv2.line(img_bgr, tuple(origin), tuple(pt_z), (255, 0, 0), 2, cv2.LINE_AA)      # z blue
#         cv2.circle(img_bgr, tuple(origin), 4, (255, 255, 255), -1)

#     return origin


# def load_background_image(ep_path: str):
#     """
#     优先读取 TAPIP3D 保存的视频第一帧；
#     如果没有，则尝试读取 rgb/0000.png 等常见路径。
#     """
#     npz_path = os.path.join(ep_path, "point_tracking", "tapip3d_result.npz")
#     if os.path.exists(npz_path):
#         data = np.load(npz_path)
#         if "video" in data:
#             img_rgb = data["video"][0]
#             return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), npz_path

#     candidates = [
#         os.path.join(ep_path, "rgb", "0000.png"),
#         os.path.join(ep_path, "rgb", "0.png"),
#         os.path.join(ep_path, "image", "0000.png"),
#         os.path.join(ep_path, "images", "0000.png"),
#         os.path.join(ep_path, "color", "0000.png"),
#     ]

#     for path in candidates:
#         if os.path.exists(path):
#             img_rgb = np.array(Image.open(path).convert("RGB"))
#             return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), path

#     return None, None


# def get_episode_path(val_dataset, local_idx: int):
#     """
#     GoalPoseSE3Dataset 里 episode_paths 已经按照 train/val split 对齐。
#     优先直接使用。
#     """
#     if hasattr(val_dataset, "episode_paths") and local_idx < len(val_dataset.episode_paths):
#         return val_dataset.episode_paths[local_idx]

#     # fallback：重新按相同 seed 切 val
#     val_episodes = []
#     for task_dir in val_dataset.data_dirs:
#         all_eps = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
#         rng = np.random.RandomState(42)
#         rng.shuffle(all_eps)
#         split_idx = int(len(all_eps) * (1 - val_dataset.val_ratio))
#         val_episodes.extend(all_eps[split_idx:])

#     if local_idx < len(val_episodes):
#         return val_episodes[local_idx]

#     return None


# def make_lang_tensor(lang_emb: torch.Tensor, device: torch.device):
#     """
#     保证语言输入形状为 [B, 1, 1024]。
#     如果是多 token [T,1024]，这里取平均，避免 encoder squeeze 维度错误。
#     """
#     if lang_emb is None:
#         return None

#     if isinstance(lang_emb, torch.Tensor):
#         arr = lang_emb.detach().cpu().numpy().astype(np.float32)
#     else:
#         arr = np.asarray(lang_emb, dtype=np.float32)

#     if arr.ndim == 1:
#         arr = arr[None, None, :]
#     elif arr.ndim == 2:
#         if arr.shape[0] != 1:
#             arr = arr.mean(axis=0, keepdims=True)
#         arr = arr[None, :, :]
#     elif arr.ndim == 3:
#         if arr.shape[1] != 1:
#             arr = arr.mean(axis=1, keepdims=True)
#     else:
#         raise ValueError(f"lang_emb shape 不合法: {arr.shape}")

#     return torch.from_numpy(arr).float().to(device)


# def build_obs_dict(sample, device: torch.device, use_lang_emb: bool):
#     """
#     GoalPoseDiffuser 需要：
#         pc_manipulated: [B, N, 3]
#         pc_target: [B, N, 3]
#         optional lang_token_embs: [B, 1, 1024]
#     """
#     obs = sample["obs"]

#     obs_dict = {
#         "pc_manipulated": obs["pc_manipulated"].float().unsqueeze(0).to(device),
#         "pc_target": obs["pc_target"].float().unsqueeze(0).to(device),
#         "agent_pos": obs["agent_pos"].float().unsqueeze(0).to(device),
#     }

#     if use_lang_emb:
#         if "lang_token_embs" in obs:
#             obs_dict["lang_token_embs"] = make_lang_tensor(obs["lang_token_embs"], device)
#         else:
#             # 防止 use_lang_emb=True 但当前样本缺语言 embedding 导致 fusion 维度不匹配
#             obs_dict["lang_token_embs"] = torch.zeros((1, 1, 1024), dtype=torch.float32, device=device)

#     return obs_dict


# def quat_rotation_error_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
#     q_pred = q_pred / max(np.linalg.norm(q_pred), 1e-8)
#     q_gt = q_gt / max(np.linalg.norm(q_gt), 1e-8)
#     dot = np.clip(abs(float(np.dot(q_pred, q_gt))), 0.0, 1.0)
#     return float(np.degrees(2.0 * np.arccos(dot)))


# def main():
#     ensure_dir(OUTPUT_DIR)

#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     print(f"[*] 加载 checkpoint: {CKPT_PATH}")
#     payload = torch.load(CKPT_PATH, pickle_module=dill, map_location="cpu")
#     cfg = payload["cfg"]

#     print("[*] 构建 validation dataset...")
#     train_dataset = hydra.utils.instantiate(cfg.task.dataset)
#     val_dataset = train_dataset.get_validation_dataset()

#     if len(val_dataset) == 0:
#         raise RuntimeError("val_dataset 为空，请检查 configs/model/task/goal_pose_multitask.yaml 的 data_dirs 和 val_ratio。")

#     print("[*] 构建 GoalPoseDiffuser...")
#     policy = hydra.utils.instantiate(cfg.policy)

#     state_dicts = payload["state_dicts"]
#     if USE_EMA and "ema_model" in state_dicts:
#         policy.load_state_dict(state_dicts["ema_model"], strict=True)
#         print("[*] 使用 ema_model 权重")
#     else:
#         policy.load_state_dict(state_dicts["model"], strict=True)
#         print("[*] 使用 model 权重")

#     if "normalizer" in payload:
#         policy.normalizer.load_state_dict(payload["normalizer"])
#         print("[*] 从 checkpoint 恢复 normalizer")
#     else:
#         normalizer = train_dataset.get_normalizer()
#         policy.set_normalizer(normalizer)
#         print("[*] checkpoint 中没有 normalizer，使用 train_dataset 重新 fit 的 normalizer")

#     policy.eval().to(device)

#     intrinsics = val_dataset.intrinsics
#     if intrinsics is None:
#         raise RuntimeError("val_dataset.intrinsics 为 None，无法把 3D 坐标系投影到图像。")

#     num_pts = int(getattr(val_dataset, "num_pts", cfg.shape_meta.obs.pc_manipulated.shape[0]))
#     use_lang_emb = bool(getattr(policy, "use_lang_emb", False))

#     print(f"[*] val episodes: {len(val_dataset)}")
#     print(f"[*] num_pts={num_pts}, use_lang_emb={use_lang_emb}, device={device}")

#     if NUM_SAMPLES >= len(val_dataset):
#         sample_indices = np.arange(len(val_dataset), dtype=int)
#     else:
#         sample_indices = np.linspace(0, len(val_dataset) - 1, NUM_SAMPLES, dtype=int)

#     rows = []

#     print(f"\n🚀 开始 GoalPose 终点位姿推理与可视化，共 {len(sample_indices)} 个样本...")

#     for viz_idx, ds_idx in enumerate(sample_indices):
#         sample = val_dataset[int(ds_idx)]
#         traj_idx = int(sample["traj_idx"].item()) if torch.is_tensor(sample["traj_idx"]) else int(sample["traj_idx"])

#         ep_path = get_episode_path(val_dataset, int(ds_idx))
#         if ep_path is None:
#             print(f"[跳过] ds_idx={ds_idx}, traj_idx={traj_idx}: 找不到 episode path")
#             continue

#         img_bgr, img_source = load_background_image(ep_path)
#         if img_bgr is None:
#             print(f"[跳过] {os.path.basename(ep_path)}: 找不到背景图")
#             continue

#         # 按 GoalPoseSE3Dataset 的逻辑重新计算 centroid_0
#         pc_man_0 = resize_points(val_dataset.pc_man_0_list[traj_idx], num_pts)
#         centroid_0 = pc_man_0.mean(axis=0).astype(np.float32)

#         obs_dict = build_obs_dict(sample, device, use_lang_emb)

#         # 为了每个样本可复现，同时避免所有样本用完全一样的初始随机序列
#         torch.manual_seed(SEED + int(viz_idx))

#         with torch.no_grad():
#             result = policy.sample_goal(obs_dict)

#         pred_pose7d = result["goal_pose7d"][0].detach().cpu().numpy().astype(np.float32)
#         pred_pose9d = result["goal_pose9d"][0].detach().cpu().numpy().astype(np.float32)
#         pred_T_goal = result["T_goal"][0].detach().cpu().numpy().astype(np.float32)

#         gt_pose7d = sample["goal_pose7d"].detach().cpu().numpy().astype(np.float32)
#         gt_pose9d = sample["goal_pose9d"].detach().cpu().numpy().astype(np.float32)

#         pred_T_cam = local_pose_to_camera_matrix(pred_pose7d, centroid_0)
#         gt_T_cam = local_pose_to_camera_matrix(gt_pose7d, centroid_0)

#         pos_err_cm = float(np.linalg.norm(pred_pose7d[:3] - gt_pose7d[:3]) * 100.0)
#         rot_err_deg = quat_rotation_error_deg(pred_pose7d[3:7], gt_pose7d[3:7])

#         print(
#             f"  [{viz_idx + 1}/{len(sample_indices)}] {os.path.basename(ep_path)} | "
#             f"pos_err={pos_err_cm:.2f} cm | rot_err={rot_err_deg:.2f} deg"
#         )

#         # ===== 绘制坐标系 =====
#         img_vis = img_bgr.copy()

#         # GT：绿色
#         draw_3d_axes_on_image(
#             img_vis,
#             gt_T_cam,
#             intrinsics,
#             centroid_ref=centroid_0,
#             scale=AXIS_LENGTH,
#             is_gt=True,
#         )

#         # Pred：RGB 三轴
#         draw_3d_axes_on_image(
#             img_vis,
#             pred_T_cam,
#             intrinsics,
#             centroid_ref=centroid_0,
#             scale=AXIS_LENGTH,
#             is_gt=False,
#         )

#         cv2.putText(
#             img_vis,
#             f"Episode: {os.path.basename(ep_path)}",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (255, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )
#         cv2.putText(
#             img_vis,
#             "GT Goal Pose: Green",
#             (20, 80),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 220, 0),
#             2,
#             cv2.LINE_AA,
#         )
#         cv2.putText(
#             img_vis,
#             "Pred Goal Pose: RGB axes",
#             (20, 120),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 165, 255),
#             2,
#             cv2.LINE_AA,
#         )
#         cv2.putText(
#             img_vis,
#             f"Err: {pos_err_cm:.2f} cm, {rot_err_deg:.2f} deg",
#             (20, 160),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (255, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )

#         ep_name = os.path.basename(ep_path)
#         out_png = os.path.join(OUTPUT_DIR, f"goal_pose_overlay_{viz_idx:02d}_{ep_name}.png")
#         out_npz = os.path.join(OUTPUT_DIR, f"goal_pose_overlay_{viz_idx:02d}_{ep_name}.npz")

#         cv2.imwrite(out_png, img_vis)

#         np.savez_compressed(
#             out_npz,
#             pred_pose7d=pred_pose7d,
#             pred_pose9d=pred_pose9d,
#             pred_T_goal=pred_T_goal,
#             pred_T_cam=pred_T_cam,
#             gt_pose7d=gt_pose7d,
#             gt_pose9d=gt_pose9d,
#             gt_T_cam=gt_T_cam,
#             centroid_0=centroid_0,
#             pos_err_cm=np.asarray(pos_err_cm, dtype=np.float32),
#             rot_err_deg=np.asarray(rot_err_deg, dtype=np.float32),
#             ep_path=ep_path,
#             img_source=img_source,
#             ckpt_path=CKPT_PATH,
#         )

#         rows.append(
#             {
#                 "viz_idx": viz_idx,
#                 "dataset_idx": int(ds_idx),
#                 "traj_idx": traj_idx,
#                 "episode": ep_name,
#                 "pos_err_cm": pos_err_cm,
#                 "rot_err_deg": rot_err_deg,
#                 "pred_x": float(pred_pose7d[0]),
#                 "pred_y": float(pred_pose7d[1]),
#                 "pred_z": float(pred_pose7d[2]),
#                 "pred_qx": float(pred_pose7d[3]),
#                 "pred_qy": float(pred_pose7d[4]),
#                 "pred_qz": float(pred_pose7d[5]),
#                 "pred_qw": float(pred_pose7d[6]),
#                 "gt_x": float(gt_pose7d[0]),
#                 "gt_y": float(gt_pose7d[1]),
#                 "gt_z": float(gt_pose7d[2]),
#                 "gt_qx": float(gt_pose7d[3]),
#                 "gt_qy": float(gt_pose7d[4]),
#                 "gt_qz": float(gt_pose7d[5]),
#                 "gt_qw": float(gt_pose7d[6]),
#                 "png_path": out_png,
#                 "npz_path": out_npz,
#             }
#         )

#     summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
#     if len(rows) > 0:
#         with open(summary_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#             writer.writeheader()
#             writer.writerows(rows)

#         mean_pos = np.mean([r["pos_err_cm"] for r in rows])
#         mean_rot = np.mean([r["rot_err_deg"] for r in rows])
#         print(f"\n📊 平均误差: {mean_pos:.2f} cm | {mean_rot:.2f} deg")
#         print(f"[*] summary 已保存: {summary_path}")

#     print(f"\n🎉 完成！结果保存在: {OUTPUT_DIR}")


# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import glob
import csv
import pathlib

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from PIL import Image

try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ================= 配置区 =================
CKPT_PATH = os.environ.get(
    "LFV_GOAL_CKPT",
    "data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest/checkpoints/latest.ckpt",
)
OUTPUT_DIR = os.environ.get("LFV_GOAL_OUTPUT_DIR", None)

NUM_SAMPLES = 5
AXIS_LENGTH = 0.05
SEED = 42
USE_EMA = True

# 重要：
# 当前 checkpoint 是用旧 GoalPoseSE3Dataset 训练的：
#   goal_pose7d = traj[-1, :7]
# 所以这里必须用 raw。
#
# 等你后续替换修正版 dataset 并重新训练后，再改成：
#   POSE_LABEL_MODE = "local"
POSE_LABEL_MODE = "local"   # choices: "raw", "local"

DEBUG_COMPARE_RAW_AND_LOCAL = True
# ==========================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resize_points(points: np.ndarray, num_pts: int) -> np.ndarray:
    """
    与 GoalPoseSE3Dataset._resize_points 保持一致。
    """
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] == num_pts:
        return points.astype(np.float32)

    if points.shape[0] > num_pts:
        idx = np.linspace(0, points.shape[0] - 1, num_pts).astype(np.int64)
        return points[idx].astype(np.float32)

    if points.shape[0] == 0:
        return np.zeros((num_pts, 3), dtype=np.float32)

    pad_idx = np.arange(num_pts - points.shape[0]) % points.shape[0]
    return np.concatenate([points, points[pad_idx]], axis=0).astype(np.float32)


def pose_to_matrix(pose7d: np.ndarray) -> np.ndarray:
    """
    pose7d: [x, y, z, qx, qy, qz, qw]
    """
    pose7d = np.asarray(pose7d, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose7d[:3]
    T[:3, :3] = R.from_quat(pose7d[3:7]).as_matrix()
    return T


def local_pose_to_camera_matrix(local_pose7d: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
    """
    local pose 转 camera transform。

    local pose 定义：
        p_local = p_cam - C0
        p_local' = R * p_local + t_local

    camera transform 定义：
        p_cam' = R * p_cam + t_cam

    因此：
        t_cam = t_local - R * C0 + C0
    """
    local_pose7d = np.asarray(local_pose7d, dtype=np.float32)
    centroid_0 = np.asarray(centroid_0, dtype=np.float32)

    R_local = R.from_quat(local_pose7d[3:7]).as_matrix()
    t_local = local_pose7d[:3]
    t_cam = t_local - R_local @ centroid_0 + centroid_0

    T_cam = np.eye(4, dtype=np.float32)
    T_cam[:3, :3] = R_local
    T_cam[:3, 3] = t_cam
    return T_cam


def pose7d_to_camera_matrix_for_overlay(pose7d: np.ndarray, centroid_0: np.ndarray, pose_mode: str) -> np.ndarray:
    """
    把模型输出/GT pose 转成用于图像 overlay 的 camera transform。

    pose_mode = "raw":
        pose7d 本身就是 camera-relative transform，即旧 dataset 当前 checkpoint 的情况。
        直接 pose_to_matrix(pose7d)。

    pose_mode = "local":
        pose7d 是 centroid-local transform，即修正版 dataset 重新训练后的情况。
        需要 local_pose_to_camera_matrix(pose7d, centroid_0)。
    """
    if pose_mode == "raw":
        return pose_to_matrix(pose7d)
    if pose_mode == "local":
        return local_pose_to_camera_matrix(pose7d, centroid_0)
    raise ValueError(f"Unknown POSE_LABEL_MODE={pose_mode}, expected 'raw' or 'local'.")


def project_points(pts_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = np.maximum(pts_3d[:, 2], 1e-5)
    x2d = pts_3d[:, 0] * fx / z + cx
    y2d = pts_3d[:, 1] * fy / z + cy

    return np.stack([x2d, y2d], axis=-1).astype(int)


def transform_point_np(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def transformed_centroid(T_cam: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
    return transform_point_np(centroid_0[None, :], T_cam)[0]


def draw_3d_axes_on_image(
    img_bgr: np.ndarray,
    T_matrix: np.ndarray,
    intrinsics: np.ndarray,
    centroid_ref: np.ndarray,
    scale: float = 0.05,
    is_gt: bool = False,
):
    """
    与你原来的轨迹可视化保持一致：

    坐标轴先锚定到第一帧 manipulated centroid：
        axes = C0 + local_axis

    然后经过 T_matrix 变换：
        axes' = T_matrix * axes

    如果 T_matrix 是 raw camera-relative transform：
        axes' = R * (C0 + axis) + t_raw

    如果 T_matrix 是从 local pose 转出来的 camera transform：
        axes' = R * (C0 + axis) + t_cam
    """
    axes_3d_local = np.array(
        [
            [0, 0, 0],
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, scale],
        ],
        dtype=np.float32,
    )

    centroid_ref = np.asarray(centroid_ref, dtype=np.float32)
    axes_3d_anchored = axes_3d_local + centroid_ref
    axes_3d_anchored_h = np.concatenate(
        [axes_3d_anchored, np.ones((4, 1), dtype=np.float32)],
        axis=1,
    )

    axes_cam_t = (T_matrix @ axes_3d_anchored_h.T).T[:, :3]
    pts_2d = project_points(axes_cam_t, intrinsics)

    origin, pt_x, pt_y, pt_z = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]

    if is_gt:
        cv2.line(img_bgr, tuple(origin), tuple(pt_x), (0, 180, 0), 1, cv2.LINE_AA)
        cv2.line(img_bgr, tuple(origin), tuple(pt_y), (0, 180, 0), 1, cv2.LINE_AA)
        cv2.line(img_bgr, tuple(origin), tuple(pt_z), (0, 180, 0), 1, cv2.LINE_AA)
        cv2.circle(img_bgr, tuple(origin), 3, (0, 230, 0), -1)
    else:
        cv2.line(img_bgr, tuple(origin), tuple(pt_x), (0, 0, 255), 2, cv2.LINE_AA)      # x red
        cv2.line(img_bgr, tuple(origin), tuple(pt_y), (0, 255, 0), 2, cv2.LINE_AA)      # y green
        cv2.line(img_bgr, tuple(origin), tuple(pt_z), (255, 0, 0), 2, cv2.LINE_AA)      # z blue
        cv2.circle(img_bgr, tuple(origin), 4, (255, 255, 255), -1)

    return origin


def load_background_image(ep_path: str):
    """
    优先读取 TAPIP3D 保存的视频第一帧；
    如果没有，则尝试读取 rgb/0000.png 等常见路径。
    """
    npz_path = os.path.join(ep_path, "point_tracking", "tapip3d_result.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        if "video" in data:
            img_rgb = data["video"][0]
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), npz_path

    candidates = [
        os.path.join(ep_path, "rgb", "0000.png"),
        os.path.join(ep_path, "rgb", "0.png"),
        os.path.join(ep_path, "image", "0000.png"),
        os.path.join(ep_path, "images", "0000.png"),
        os.path.join(ep_path, "color", "0000.png"),
    ]

    for path in candidates:
        if os.path.exists(path):
            img_rgb = np.array(Image.open(path).convert("RGB"))
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), path

    return None, None


def get_episode_path(val_dataset, local_idx: int):
    """
    GoalPoseSE3Dataset.episode_paths 已经按 train/val split 对齐。
    优先直接使用。
    """
    if hasattr(val_dataset, "episode_paths") and local_idx < len(val_dataset.episode_paths):
        return val_dataset.episode_paths[local_idx]

    val_episodes = []
    for task_dir in val_dataset.data_dirs:
        all_eps = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
        rng = np.random.RandomState(42)
        rng.shuffle(all_eps)
        split_idx = int(len(all_eps) * (1 - val_dataset.val_ratio))
        val_episodes.extend(all_eps[split_idx:])

    if local_idx < len(val_episodes):
        return val_episodes[local_idx]

    return None


def make_lang_tensor(lang_emb, device: torch.device):
    """
    保证语言输入形状为 [B, 1, 1024]。

    如果是多 token [T,1024]，这里取平均成 [1,1024]，
    避免 encoder 中 lang_proj 后出现 [B,T,128] 导致 fusion 维度错误。
    """
    if lang_emb is None:
        return None

    if isinstance(lang_emb, torch.Tensor):
        arr = lang_emb.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(lang_emb, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        if arr.shape[0] != 1:
            arr = arr.mean(axis=0, keepdims=True)
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[1] != 1:
            arr = arr.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"lang_emb shape 不合法: {arr.shape}")

    return torch.from_numpy(arr).float().to(device)


def build_obs_dict(sample, device: torch.device, use_lang_emb: bool):
    """
    GoalPoseDiffuser 需要：
        pc_manipulated: [B, N, 3]
        pc_target: [B, N, 3]
        optional lang_token_embs: [B, 1, 1024]
    """
    obs = sample["obs"]

    obs_dict = {
        "pc_manipulated": obs["pc_manipulated"].float().unsqueeze(0).to(device),
        "pc_target": obs["pc_target"].float().unsqueeze(0).to(device),
        "agent_pos": obs["agent_pos"].float().unsqueeze(0).to(device),
    }

    if use_lang_emb:
        if "lang_token_embs" in obs:
            obs_dict["lang_token_embs"] = make_lang_tensor(obs["lang_token_embs"], device)
        else:
            obs_dict["lang_token_embs"] = torch.zeros((1, 1, 1024), dtype=torch.float32, device=device)

    return obs_dict


def quat_rotation_error_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
    q_pred = q_pred / max(np.linalg.norm(q_pred), 1e-8)
    q_gt = q_gt / max(np.linalg.norm(q_gt), 1e-8)
    dot = np.clip(abs(float(np.dot(q_pred, q_gt))), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)))


def debug_compare_raw_and_local_projection(gt_pose7d, centroid_0, intrinsics):
    """
    用于验证你这次的问题：

    raw 正确画法：
        T_raw = pose_to_matrix(gt_pose7d)
        origin = T_raw * C0

    错误画法：
        T_wrong = local_pose_to_camera_matrix(gt_pose7d, C0)
        origin = T_wrong * C0

    如果 raw 落到盘子上，而 wrong 落到图 1 的绿色位置，
    就说明之前确实把 raw pose 错当成 local pose 了。
    """
    T_raw = pose_to_matrix(gt_pose7d)
    T_wrong = local_pose_to_camera_matrix(gt_pose7d, centroid_0)

    p_raw = transformed_centroid(T_raw, centroid_0)
    p_wrong = transformed_centroid(T_wrong, centroid_0)

    uv_raw = project_points(p_raw[None, :], intrinsics)[0]
    uv_wrong = project_points(p_wrong[None, :], intrinsics)[0]

    return {
        "p_raw": p_raw,
        "p_wrong": p_wrong,
        "uv_raw": uv_raw,
        "uv_wrong": uv_wrong,
    }


def main():
    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = os.path.join(
            str(pathlib.Path(CKPT_PATH).resolve().parents[1]),
            "goal_pose_val_overlay",
        )
    ensure_dir(output_dir)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[*] 加载 checkpoint: {CKPT_PATH}")
    payload = torch.load(CKPT_PATH, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    print("[*] 构建 validation dataset...")
    train_dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = train_dataset.get_validation_dataset()

    if len(val_dataset) == 0:
        raise RuntimeError("val_dataset 为空，请检查 configs/model/task/goal_pose_multitask.yaml 的 data_dirs 和 val_ratio。")

    print("[*] 构建 GoalPoseDiffuser...")
    policy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload["state_dicts"]
    if USE_EMA and "ema_model" in state_dicts:
        policy.load_state_dict(state_dicts["ema_model"], strict=True)
        print("[*] 使用 ema_model 权重")
    else:
        policy.load_state_dict(state_dicts["model"], strict=True)
        print("[*] 使用 model 权重")

    if "normalizer" in payload:
        policy.normalizer.load_state_dict(payload["normalizer"])
        print("[*] 从 checkpoint 恢复 normalizer")
    else:
        normalizer = train_dataset.get_normalizer()
        policy.set_normalizer(normalizer)
        print("[*] checkpoint 中没有 normalizer，使用 train_dataset 重新 fit 的 normalizer")

    policy.eval().to(device)

    intrinsics = val_dataset.intrinsics
    if intrinsics is None:
        raise RuntimeError("val_dataset.intrinsics 为 None，无法把 3D 坐标系投影到图像。")

    num_pts = int(getattr(val_dataset, "num_pts", cfg.shape_meta.obs.pc_manipulated.shape[0]))
    use_lang_emb = bool(getattr(policy, "use_lang_emb", False))

    print(f"[*] val episodes: {len(val_dataset)}")
    print(f"[*] num_pts={num_pts}, use_lang_emb={use_lang_emb}, device={device}")
    print(f"[*] POSE_LABEL_MODE={POSE_LABEL_MODE}")

    if NUM_SAMPLES >= len(val_dataset):
        sample_indices = np.arange(len(val_dataset), dtype=int)
    else:
        sample_indices = np.linspace(0, len(val_dataset) - 1, NUM_SAMPLES, dtype=int)

    rows = []

    print(f"\n🚀 开始 GoalPose 终点位姿推理与可视化，共 {len(sample_indices)} 个样本...")

    for viz_idx, ds_idx in enumerate(sample_indices):
        ds_idx = int(ds_idx)
        sample = val_dataset[ds_idx]

        traj_idx = int(sample["traj_idx"].item()) if torch.is_tensor(sample["traj_idx"]) else int(sample["traj_idx"])

        ep_path = get_episode_path(val_dataset, ds_idx)
        if ep_path is None:
            print(f"[跳过] ds_idx={ds_idx}, traj_idx={traj_idx}: 找不到 episode path")
            continue

        img_bgr, img_source = load_background_image(ep_path)
        if img_bgr is None:
            print(f"[跳过] {os.path.basename(ep_path)}: 找不到背景图")
            continue

        # 按 GoalPoseSE3Dataset 的逻辑重新计算 centroid_0。
        # 注意：必须 resize 后再 mean，和 dataset __getitem__ 对齐。
        pc_man_0 = resize_points(val_dataset.pc_man_0_list[traj_idx], num_pts)
        centroid_0 = pc_man_0.mean(axis=0).astype(np.float32)

        obs_dict = build_obs_dict(sample, device, use_lang_emb)

        # 每个样本固定不同 seed，保证可复现但不是所有样本同一条随机噪声。
        torch.manual_seed(SEED + viz_idx)

        with torch.no_grad():
            result = policy.sample_goal(obs_dict)

        pred_pose7d = result["goal_pose7d"][0].detach().cpu().numpy().astype(np.float32)
        pred_pose9d = result["goal_pose9d"][0].detach().cpu().numpy().astype(np.float32)
        pred_T_goal = result["T_goal"][0].detach().cpu().numpy().astype(np.float32)

        gt_pose7d = sample["goal_pose7d"].detach().cpu().numpy().astype(np.float32)
        gt_pose9d = sample["goal_pose9d"].detach().cpu().numpy().astype(np.float32)

        pred_T_cam = pose7d_to_camera_matrix_for_overlay(pred_pose7d, centroid_0, POSE_LABEL_MODE)
        gt_T_cam = pose7d_to_camera_matrix_for_overlay(gt_pose7d, centroid_0, POSE_LABEL_MODE)

        # label 空间误差：和训练日志一致。
        label_pos_err_cm = float(np.linalg.norm(pred_pose7d[:3] - gt_pose7d[:3]) * 100.0)
        rot_err_deg = quat_rotation_error_deg(pred_pose7d[3:7], gt_pose7d[3:7])

        # 图像物理终点误差：比较 transformed centroid 的 camera 坐标。
        pred_centroid_cam = transformed_centroid(pred_T_cam, centroid_0)
        gt_centroid_cam = transformed_centroid(gt_T_cam, centroid_0)
        centroid_pos_err_cm = float(np.linalg.norm(pred_centroid_cam - gt_centroid_cam) * 100.0)

        ep_name = os.path.basename(ep_path)

        print(
            f"  [{viz_idx + 1}/{len(sample_indices)}] {ep_name} | "
            f"label_pos_err={label_pos_err_cm:.2f} cm | "
            f"centroid_pos_err={centroid_pos_err_cm:.2f} cm | "
            f"rot_err={rot_err_deg:.2f} deg"
        )

        if DEBUG_COMPARE_RAW_AND_LOCAL and POSE_LABEL_MODE == "raw" and viz_idx == 0:
            dbg = debug_compare_raw_and_local_projection(gt_pose7d, centroid_0, intrinsics)
            print("  [Debug] GT raw projection uv:", dbg["uv_raw"], "p:", dbg["p_raw"])
            print("  [Debug] GT wrong-local projection uv:", dbg["uv_wrong"], "p:", dbg["p_wrong"])
            print("  [Debug] 如果 raw uv 在正确终点、wrong uv 在错误位置，则说明旧脚本的问题已确认。")

        # ===== 绘制坐标系 =====
        img_vis = img_bgr.copy()

        draw_3d_axes_on_image(
            img_vis,
            gt_T_cam,
            intrinsics,
            centroid_ref=centroid_0,
            scale=AXIS_LENGTH,
            is_gt=True,
        )

        draw_3d_axes_on_image(
            img_vis,
            pred_T_cam,
            intrinsics,
            centroid_ref=centroid_0,
            scale=AXIS_LENGTH,
            is_gt=False,
        )

        cv2.putText(
            img_vis,
            f"Episode: {ep_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_vis,
            "GT Goal Pose: Green",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_vis,
            "Pred Goal Pose: RGB axes",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_vis,
            f"Err centroid: {centroid_pos_err_cm:.2f} cm, rot: {rot_err_deg:.2f} deg",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_png = os.path.join(output_dir, f"goal_pose_overlay_{viz_idx:02d}_{ep_name}.png")
        out_npz = os.path.join(output_dir, f"goal_pose_overlay_{viz_idx:02d}_{ep_name}.npz")

        cv2.imwrite(out_png, img_vis)

        np.savez_compressed(
            out_npz,
            pred_pose7d=pred_pose7d,
            pred_pose9d=pred_pose9d,
            pred_T_goal=pred_T_goal,
            pred_T_cam=pred_T_cam,
            pred_centroid_cam=pred_centroid_cam,
            gt_pose7d=gt_pose7d,
            gt_pose9d=gt_pose9d,
            gt_T_cam=gt_T_cam,
            gt_centroid_cam=gt_centroid_cam,
            centroid_0=centroid_0,
            label_pos_err_cm=np.asarray(label_pos_err_cm, dtype=np.float32),
            centroid_pos_err_cm=np.asarray(centroid_pos_err_cm, dtype=np.float32),
            rot_err_deg=np.asarray(rot_err_deg, dtype=np.float32),
            ep_path=ep_path,
            img_source=img_source,
            ckpt_path=CKPT_PATH,
            pose_label_mode=POSE_LABEL_MODE,
        )

        rows.append(
            {
                "viz_idx": viz_idx,
                "dataset_idx": ds_idx,
                "traj_idx": traj_idx,
                "episode": ep_name,
                "label_pos_err_cm": label_pos_err_cm,
                "centroid_pos_err_cm": centroid_pos_err_cm,
                "rot_err_deg": rot_err_deg,
                "pred_x": float(pred_pose7d[0]),
                "pred_y": float(pred_pose7d[1]),
                "pred_z": float(pred_pose7d[2]),
                "pred_qx": float(pred_pose7d[3]),
                "pred_qy": float(pred_pose7d[4]),
                "pred_qz": float(pred_pose7d[5]),
                "pred_qw": float(pred_pose7d[6]),
                "gt_x": float(gt_pose7d[0]),
                "gt_y": float(gt_pose7d[1]),
                "gt_z": float(gt_pose7d[2]),
                "gt_qx": float(gt_pose7d[3]),
                "gt_qy": float(gt_pose7d[4]),
                "gt_qz": float(gt_pose7d[5]),
                "gt_qw": float(gt_pose7d[6]),
                "pred_centroid_x": float(pred_centroid_cam[0]),
                "pred_centroid_y": float(pred_centroid_cam[1]),
                "pred_centroid_z": float(pred_centroid_cam[2]),
                "gt_centroid_x": float(gt_centroid_cam[0]),
                "gt_centroid_y": float(gt_centroid_cam[1]),
                "gt_centroid_z": float(gt_centroid_cam[2]),
                "png_path": out_png,
                "npz_path": out_npz,
                "ep_path": ep_path,
                "img_source": img_source,
                "pose_label_mode": POSE_LABEL_MODE,
            }
        )

    summary_path = os.path.join(output_dir, "summary.csv")
    if len(rows) > 0:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        mean_label_pos = np.mean([r["label_pos_err_cm"] for r in rows])
        mean_centroid_pos = np.mean([r["centroid_pos_err_cm"] for r in rows])
        mean_rot = np.mean([r["rot_err_deg"] for r in rows])

        print(f"\n📊 平均 label 平移误差: {mean_label_pos:.2f} cm")
        print(f"📊 平均 centroid 终点误差: {mean_centroid_pos:.2f} cm")
        print(f"📊 平均旋转误差: {mean_rot:.2f} deg")
        print(f"[*] summary 已保存: {summary_path}")

    print(f"\n🎉 完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
