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
import json
import csv
import pathlib

import cv2
import dill
import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass


OmegaConf.register_new_resolver("eval", eval, replace=True)


# ================= 配置区 =================

SECOND_STAGE_CKPT_PATH = (
    os.environ.get(
        "LFV_FULL64_CKPT",
        "data/outputs/full64/pickNplace_lfv_full64_seed42/latest/checkpoints/latest.ckpt",
    )
)

SCENE_ROOT = os.environ.get("LFV_SCENE_ROOT", "data/env_data/pickNplace_lfv")

FIRST_STAGE_OUTPUT_SUBDIR = "model_inference_goal_pose"
FIRST_STAGE_GOAL_POSE7D_NAME = "pred_goal_pose7d.npy"

OUTPUT_SUBDIR = "model_inference_full64_traj"

LANG_EMB_PATH = os.path.join(SCENE_ROOT, "lang_emb.npy")

INTRINSICS_SOURCE = "depth_intrinsics_original"

IMAGE_NAME = "0000.png"
DEPTH_NAME = "0000.npy"

NUM_PTS_OVERRIDE = 64

SEED = 42
MAX_SCENES = 5
USE_EMA = True

AXIS_LENGTH = 0.05
DRAW_EVERY = 4
DRAW_SAMPLE_POINTS = True

# 第一阶段输出的 pred_goal_pose7d 是否已经是 local pose。
# 你现在第一阶段脚本 POSE_LABEL_MODE="local"，所以这里必须保持 "local"。
# 如果以后第一阶段输出 raw camera pose，再改成 "raw"。
FIRST_STAGE_GOAL_MODE = "local"  # "local" or "raw"

AFFORDANCE_POINTS_CANDIDATES = [
    os.path.join("affordance_sample_points", "sample_points.npy"),
    os.path.join("affordance_sample_points", "sampled_2d_uniform.npy"),
    os.path.join("affordance_sample_points", "affordance_sampled_2d_uniform.npy"),
]

TARGET_POINTS_CANDIDATES = [
    os.path.join("target_sample_points", "sample_points.npy"),
    os.path.join("target_sample_points", "target_sampled_2d_uniform.npy"),
    os.path.join("target_sample_points", "sampled_2d_uniform.npy"),
]

# ==========================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    print("[Warn] CUDA 不可用，使用 CPU")
    return torch.device("cpu")


def intrinsics_dict_to_matrix(intrinsics_raw: dict) -> np.ndarray:
    if isinstance(intrinsics_raw, dict):
        fx = intrinsics_raw.get("fx", intrinsics_raw.get("f"))
        fy = intrinsics_raw.get("fy", intrinsics_raw.get("f"))
        cx = intrinsics_raw.get("cx", intrinsics_raw.get("ppx"))
        cy = intrinsics_raw.get("cy", intrinsics_raw.get("ppy"))
        if None in (fx, fy, cx, cy):
            raise ValueError(f"无法解析相机内参: {intrinsics_raw}")
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float32,
        )

    return np.asarray(intrinsics_raw, dtype=np.float32)


def load_scene_camera_params(scene_dir: str, intrinsics_source: str = INTRINSICS_SOURCE):
    meta_path = os.path.join(scene_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到 scene/episode meta.json: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if intrinsics_source not in meta:
        raise KeyError(
            f"{meta_path} 中没有 {intrinsics_source}; 可用字段: {sorted(meta.keys())}"
        )

    intrinsics = intrinsics_dict_to_matrix(meta[intrinsics_source])
    depth_scale = float(meta.get("depth_scale", 1.0))
    return intrinsics, depth_scale


def load_rgb(scene_dir: str) -> np.ndarray:
    candidates = [
        os.path.join(scene_dir, "rgb", IMAGE_NAME),
        os.path.join(scene_dir, "rgb", "0.png"),
        os.path.join(scene_dir, "image", IMAGE_NAME),
        os.path.join(scene_dir, "images", IMAGE_NAME),
        os.path.join(scene_dir, "color", IMAGE_NAME),
    ]

    for path in candidates:
        if os.path.exists(path):
            return np.array(Image.open(path).convert("RGB"))

    raise FileNotFoundError(f"未找到 RGB 图像，已尝试: {candidates}")


def load_depth(scene_dir: str) -> np.ndarray:
    candidates = [
        os.path.join(scene_dir, "depth", DEPTH_NAME),
        os.path.join(scene_dir, "depth", "0000.npz"),
        os.path.join(scene_dir, "depth", "0.npy"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = "depth" if "depth" in arr else arr.files[0]
            arr = arr[key]

        return np.asarray(arr, dtype=np.float32)

    raise FileNotFoundError(f"未找到 depth 文件，已尝试: {candidates}")


def load_points_2d_from_candidates(scene_dir: str, candidates):
    """
    兼容两种格式：
    1. ndarray: [N, 2]
    2. dict npy: {"query_points_2d": ...}
    """
    tried = []

    for rel_path in candidates:
        full_path = os.path.join(scene_dir, rel_path)
        tried.append(full_path)

        if not os.path.exists(full_path):
            continue

        arr = np.load(full_path, allow_pickle=True)

        try:
            obj = arr.item()
            if isinstance(obj, dict) and "query_points_2d" in obj:
                pts = np.asarray(obj["query_points_2d"], dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    return pts, full_path
        except Exception:
            pass

        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float32), full_path

    raise FileNotFoundError(f"未找到可用 2D 点文件，已尝试: {tried}")


def unproject_2d_to_3d(
    pts_2d: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    target_num: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    pts_3d = []

    for p in pts_2d:
        x, y = int(p[0]), int(p[1])

        if x < 0 or y < 0 or x >= depth_map.shape[1] or y >= depth_map.shape[0]:
            continue

        z = float(depth_map[y, x])
        if z <= 0 or np.isnan(z) or not np.isfinite(z):
            continue

        x_c = (x - cx) * z / fx
        y_c = (y - cy) * z / fy
        pts_3d.append([x_c, y_c, z])

    pts_3d = np.asarray(pts_3d, dtype=np.float32)

    if len(pts_3d) == 0:
        return np.zeros((target_num, 3), dtype=np.float32)

    if len(pts_3d) > target_num:
        idx = rng.choice(len(pts_3d), target_num, replace=False)
        pts_3d = pts_3d[idx]
    elif len(pts_3d) < target_num:
        pad_idx = rng.choice(len(pts_3d), target_num - len(pts_3d), replace=True)
        pts_3d = np.vstack([pts_3d, pts_3d[pad_idx]])

    return pts_3d.astype(np.float32)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return (q / norm).astype(np.float32)


def pose_to_matrix(pose7d: np.ndarray) -> np.ndarray:
    pose7d = np.asarray(pose7d, dtype=np.float32)
    quat = normalize_quat(pose7d[3:7])

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose7d[:3]
    T[:3, :3] = R.from_quat(quat).as_matrix().astype(np.float32)
    return T


def matrix_to_pose7d(T: np.ndarray) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = T[:3, 3].astype(np.float32)
    pose[3:7] = R.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
    pose[3:7] = normalize_quat(pose[3:7])
    return pose


def matrix_to_pose9d(T: np.ndarray) -> np.ndarray:
    """
    与 diffusion_policy_3d.model.goal.pose_utils.matrix_to_pose9d 的 rot6d 顺序保持一致：
        rot6d = R[:, :2].T.reshape(6)
    即先第一列，再第二列。
    """
    pose9d = np.zeros(9, dtype=np.float32)
    pose9d[:3] = T[:3, 3].astype(np.float32)
    pose9d[3:9] = T[:3, :3][:, :2].T.reshape(6).astype(np.float32)
    return pose9d


def pose7d_to_pose9d(pose7d: np.ndarray) -> np.ndarray:
    return matrix_to_pose9d(pose_to_matrix(pose7d))


def raw_pose7d_to_local_pose7d(raw_pose7d: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
    """
    raw camera pose -> centroid-local pose.

    raw pose:
        p_cam' = R @ p_cam + t_raw

    local:
        p_local = p_cam - C0
        p_local' = R @ p_local + t_local

    所以:
        t_local = R @ C0 + t_raw - C0
    """
    raw_pose7d = np.asarray(raw_pose7d, dtype=np.float32).copy()
    centroid_0 = np.asarray(centroid_0, dtype=np.float32).reshape(3)

    quat = normalize_quat(raw_pose7d[3:7])
    R_raw = R.from_quat(quat).as_matrix().astype(np.float32)
    t_raw = raw_pose7d[:3].astype(np.float32)

    local_pose = raw_pose7d[:7].copy()
    local_pose[:3] = R_raw @ centroid_0 + t_raw - centroid_0
    local_pose[3:7] = quat
    return local_pose.astype(np.float32)


def local_pose_to_camera_matrix(local_pose7d: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
    """
    local pose -> camera-space transform.

    local:
        t_local = R @ C0 + t_cam - C0

    camera:
        t_cam = t_local - R @ C0 + C0
    """
    local_pose7d = np.asarray(local_pose7d, dtype=np.float32)
    quat = normalize_quat(local_pose7d[3:7])
    R_local = R.from_quat(quat).as_matrix().astype(np.float32)
    t_local = local_pose7d[:3].astype(np.float32)

    t_cam = t_local - R_local @ centroid_0 + centroid_0

    T_cam = np.eye(4, dtype=np.float32)
    T_cam[:3, :3] = R_local
    T_cam[:3, 3] = t_cam
    return T_cam


def compose_pose(base_pose7d: np.ndarray, rel_pose7d: np.ndarray) -> np.ndarray:
    """
    base_pose7d: absolute local pose
    rel_pose7d: residual pose relative to base
    return: absolute local pose
    """
    T_base = pose_to_matrix(base_pose7d)
    T_rel = pose_to_matrix(rel_pose7d)
    T_abs = T_base @ T_rel
    return matrix_to_pose7d(T_abs)


def project_points(pts_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = np.maximum(pts_3d[:, 2], 1e-6)
    u = pts_3d[:, 0] * fx / z + cx
    v = pts_3d[:, 1] * fy / z + cy

    return np.stack([u, v], axis=-1).astype(np.int32)


def draw_points(img_bgr: np.ndarray, pts_2d: np.ndarray, color=(0, 255, 255), radius=2):
    h, w = img_bgr.shape[:2]
    for p in pts_2d:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img_bgr, (x, y), radius, color, -1)


def draw_3d_axes_on_image(
    img_bgr: np.ndarray,
    T_camera: np.ndarray,
    intrinsics: np.ndarray,
    centroid_ref: np.ndarray,
    scale: float = 0.05,
    mode: str = "pred",
):
    """
    与你之前的可视化逻辑保持一致：
    坐标轴先锚定在第一帧 manipulated centroid_ref 上，再经过 T_camera 变换。
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
    elif mode == "goal":
        color_x = color_y = color_z = (0, 180, 255)
        circle_color = (0, 180, 255)
        thickness = 3
    elif mode == "traj_sparse":
        color_x = (0, 0, 180)
        color_y = (0, 180, 0)
        color_z = (180, 0, 0)
        circle_color = (255, 255, 255)
        thickness = 1
    else:
        color_x = (0, 0, 255)
        color_y = (0, 255, 0)
        color_z = (255, 0, 0)
        circle_color = (255, 255, 255)
        thickness = 2

    cv2.line(img_bgr, tuple(origin), tuple(pt_x), color_x, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, tuple(origin), tuple(pt_y), color_y, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, tuple(origin), tuple(pt_z), color_z, thickness, cv2.LINE_AA)
    cv2.circle(img_bgr, tuple(origin), 3, circle_color, -1)

    return origin


def draw_polyline(img_bgr: np.ndarray, points_2d, color=(0, 165, 255), thickness=2):
    if len(points_2d) < 2:
        return

    h, w = img_bgr.shape[:2]

    for p0, p1 in zip(points_2d[:-1], points_2d[1:]):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])

        # 两端至少有一个在图像附近，就尝试绘制。
        if (
            (-w <= x0 <= 2 * w and -h <= y0 <= 2 * h)
            or (-w <= x1 <= 2 * w and -h <= y1 <= 2 * h)
        ):
            cv2.line(img_bgr, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def normalize_lang_emb_shape(emb: np.ndarray) -> np.ndarray:
    """
    输出统一为 [1, 1024]。
    如果是 [T,1024]，做 mean pooling。
    """
    emb = np.asarray(emb, dtype=np.float32)

    if emb.ndim == 1:
        emb = emb[None, :]
    elif emb.ndim == 2:
        if emb.shape[0] != 1:
            emb = emb.mean(axis=0, keepdims=True)
    elif emb.ndim == 3:
        emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"lang_emb 维度不支持: {emb.shape}")

    if emb.shape[-1] != 1024:
        raise ValueError(f"lang_emb 最后一维应为 1024，实际为 {emb.shape}")

    return emb.astype(np.float32)


def try_load_lang_emb(scene_root: str, task_data_dirs, require_lang: bool):
    if not require_lang:
        return None, None

    if os.path.exists(LANG_EMB_PATH):
        emb = np.load(LANG_EMB_PATH).astype(np.float32)
        return normalize_lang_emb_shape(emb), LANG_EMB_PATH

    for data_dir in task_data_dirs:
        lang_path = os.path.join(str(data_dir), "lang_emb.npy")
        if os.path.exists(lang_path):
            emb = np.load(lang_path).astype(np.float32)
            return normalize_lang_emb_shape(emb), lang_path

    print("[Warn] 未找到 lang_emb.npy，使用 zero fallback: [1,1024]")
    return np.zeros((1, 1024), dtype=np.float32), "zero_fallback"


def load_first_stage_goal_pose7d(scene_dir: str):
    pose_path = os.path.join(
        scene_dir,
        FIRST_STAGE_OUTPUT_SUBDIR,
        FIRST_STAGE_GOAL_POSE7D_NAME,
    )

    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"未找到第一阶段预测终点位姿: {pose_path}")

    pose7d = np.load(pose_path).astype(np.float32).reshape(-1)
    if pose7d.shape[0] < 7:
        raise ValueError(f"第一阶段 pred_goal_pose7d 维度错误: {pose_path}, shape={pose7d.shape}")

    pose7d = pose7d[:7].copy()
    pose7d[3:7] = normalize_quat(pose7d[3:7])
    return pose7d, pose_path


def build_second_stage_obs_dict(
    pc_man_0_local: np.ndarray,
    pc_tgt_0_local: np.ndarray,
    start_pose7d_local: np.ndarray,
    goal_pose7d_local: np.ndarray,
    lang_emb: np.ndarray,
    device: torch.device,
):
    """
    第二阶段 Full64 policy 输入：

    obs["pc_manipulated"]:    [B, 1, N, 3]
    obs["pc_target"]:         [B, 1, N, 3]
    obs["agent_pos"]:         [B, 1, 7]
    obs["goal_pose9d"]:       [B, 1, 9]
    obs["goal_delta_pose9d"]: [B, 1, 9]
    obs["goal_delta_pose7d"]: [B, 1, 7]
    obs["lang_token_embs"]:   [B, 1, 1024]
    """
    T_start = pose_to_matrix(start_pose7d_local)
    T_goal = pose_to_matrix(goal_pose7d_local)
    T_delta = np.linalg.inv(T_start) @ T_goal

    goal_pose9d = matrix_to_pose9d(T_goal)
    goal_delta_pose7d = matrix_to_pose7d(T_delta)
    goal_delta_pose9d = matrix_to_pose9d(T_delta)

    obs_dict = {
        "pc_manipulated": torch.from_numpy(pc_man_0_local)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),

        "pc_target": torch.from_numpy(pc_tgt_0_local)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),

        "agent_pos": torch.from_numpy(start_pose7d_local)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),

        "goal_pose9d": torch.from_numpy(goal_pose9d)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),

        "goal_delta_pose9d": torch.from_numpy(goal_delta_pose9d)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),

        "goal_delta_pose7d": torch.from_numpy(goal_delta_pose7d)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),
    }

    if lang_emb is not None:
        obs_dict["lang_token_embs"] = torch.from_numpy(lang_emb).float().unsqueeze(0).to(device)

    debug_goal = {
        "goal_pose7d_local": goal_pose7d_local,
        "goal_pose9d": goal_pose9d,
        "goal_delta_pose7d": goal_delta_pose7d,
        "goal_delta_pose9d": goal_delta_pose9d,
        "T_start": T_start,
        "T_goal": T_goal,
        "T_delta": T_delta,
    }

    return obs_dict, debug_goal


def load_second_stage_policy(device: torch.device):
    print(f"[*] 加载第二阶段 checkpoint: {SECOND_STAGE_CKPT_PATH}")
    payload = torch.load(SECOND_STAGE_CKPT_PATH, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    print("[*] 构建第二阶段 Full64 policy...")
    policy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload.get("state_dicts", {})
    if USE_EMA and "ema_model" in state_dicts:
        policy.load_state_dict(state_dicts["ema_model"], strict=True)
        print("[*] 使用 ema_model 权重")
    else:
        policy.load_state_dict(state_dicts["model"], strict=True)
        print("[*] 使用 model 权重")

    # 当前 TrainDP3Workspace 的 checkpoint 通常已经把 normalizer 保存在 model/ema_model state_dict 里。
    # 如果发现旧 checkpoint 没带 normalizer，可退回重新 fit。
    try:
        _ = policy.normalizer
    except Exception:
        print("[Warn] policy 中未发现 normalizer，尝试从训练 dataset 重新 fit")
        train_dataset = hydra.utils.instantiate(cfg.task.dataset)
        policy.set_normalizer(train_dataset.get_normalizer())

    policy.eval().to(device)
    return policy, cfg


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)

    device = choose_device()

    policy, cfg = load_second_stage_policy(device)

    task_data_dirs = OmegaConf.to_container(cfg.task.dataset.data_dirs, resolve=True)
    if isinstance(task_data_dirs, str):
        task_data_dirs = [task_data_dirs]

    if NUM_PTS_OVERRIDE is not None:
        num_pts = int(NUM_PTS_OVERRIDE)
    else:
        num_pts = int(getattr(cfg.task.dataset, "num_pts", 64))

    require_lang = bool(getattr(cfg.policy, "use_lang_emb", False))
    lang_emb, lang_emb_source = try_load_lang_emb(SCENE_ROOT, task_data_dirs, require_lang)

    scene_dirs = sorted(
        d for d in glob.glob(os.path.join(SCENE_ROOT, "scene_*"))
        if os.path.isdir(d)
    )

    if MAX_SCENES is not None:
        scene_dirs = scene_dirs[:MAX_SCENES]

    print(f"[*] device={device}")
    print(f"[*] scene_root={SCENE_ROOT}")
    print(f"[*] scenes={len(scene_dirs)}")
    print(f"[*] num_pts={num_pts}")
    print(f"[*] first_stage_goal_mode={FIRST_STAGE_GOAL_MODE}")
    print(f"[*] use_lang_emb={require_lang}, lang_emb_source={lang_emb_source}")

    summary_rows = []

    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene_name = os.path.basename(scene_dir)
        print(f"\n--- [{scene_idx + 1}/{len(scene_dirs)}] 推理 {scene_name} ---")

        try:
            rgb = load_rgb(scene_dir)
            depth = load_depth(scene_dir)
            intrinsics, depth_scale = load_scene_camera_params(scene_dir, INTRINSICS_SOURCE)
            depth = depth.astype(np.float32) * depth_scale

            affordance_pts_2d, affordance_pts_path = load_points_2d_from_candidates(
                scene_dir,
                AFFORDANCE_POINTS_CANDIDATES,
            )
            target_pts_2d, target_pts_path = load_points_2d_from_candidates(
                scene_dir,
                TARGET_POINTS_CANDIDATES,
            )

            pc_man_0 = unproject_2d_to_3d(
                pts_2d=affordance_pts_2d,
                depth_map=depth,
                intrinsics=intrinsics,
                target_num=num_pts,
                rng=rng,
            )
            pc_tgt_0 = unproject_2d_to_3d(
                pts_2d=target_pts_2d,
                depth_map=depth,
                intrinsics=intrinsics,
                target_num=num_pts,
                rng=rng,
            )

            centroid_0 = pc_man_0.mean(axis=0).astype(np.float32)

            pc_man_0_local = (pc_man_0 - centroid_0).astype(np.float32)
            pc_tgt_0_local = (pc_tgt_0 - centroid_0).astype(np.float32)

            first_stage_goal_pose7d, first_goal_path = load_first_stage_goal_pose7d(scene_dir)

            if FIRST_STAGE_GOAL_MODE == "local":
                goal_pose7d_local = first_stage_goal_pose7d.copy()
            elif FIRST_STAGE_GOAL_MODE == "raw":
                goal_pose7d_local = raw_pose7d_to_local_pose7d(first_stage_goal_pose7d, centroid_0)
            else:
                raise ValueError(f"FIRST_STAGE_GOAL_MODE 只能是 local/raw，当前为: {FIRST_STAGE_GOAL_MODE}")

            # Full64 从初始帧生成完整 residual trajectory。初始 local pose 设为 identity。
            start_pose7d_local = np.zeros(7, dtype=np.float32)
            start_pose7d_local[6] = 1.0

            obs_dict, debug_goal = build_second_stage_obs_dict(
                pc_man_0_local=pc_man_0_local,
                pc_tgt_0_local=pc_tgt_0_local,
                start_pose7d_local=start_pose7d_local,
                goal_pose7d_local=goal_pose7d_local,
                lang_emb=lang_emb,
                device=device,
            )

            torch.manual_seed(SEED + scene_idx)

            with torch.no_grad():
                result = policy.predict_action(obs_dict)

            pred_residual_actions = result.get("action_pred", result["action"])[0]
            pred_residual_actions = pred_residual_actions.detach().cpu().numpy().astype(np.float32)

            # residual action -> absolute local pose -> camera matrix
            pred_local_poses = []
            pred_camera_matrices = []
            pred_origins_2d = []

            for i in range(pred_residual_actions.shape[0]):
                rel_pose = pred_residual_actions[i, :7]
                abs_local_pose = compose_pose(start_pose7d_local, rel_pose)

                # 四元数符号连续性，避免可视化时方向跳变
                if len(pred_local_poses) > 0:
                    if np.dot(abs_local_pose[3:7], pred_local_poses[-1][3:7]) < 0:
                        abs_local_pose[3:7] *= -1

                T_cam = local_pose_to_camera_matrix(abs_local_pose, centroid_0)

                pred_local_poses.append(abs_local_pose)
                pred_camera_matrices.append(T_cam)

            pred_local_poses = np.stack(pred_local_poses, axis=0).astype(np.float32)
            pred_camera_matrices = np.stack(pred_camera_matrices, axis=0).astype(np.float32)

            print(f"  centroid_0={centroid_0}")
            print(f"  first_goal_path={first_goal_path}")
            print(f"  goal_pose7d_local={goal_pose7d_local}")
            print(f"  goal_delta_pose7d={debug_goal['goal_delta_pose7d']}")
            print(f"  pred_residual_actions shape={pred_residual_actions.shape}")

            # ================= 可视化 =================
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if DRAW_SAMPLE_POINTS:
                draw_points(img_bgr, affordance_pts_2d, color=(0, 255, 255), radius=2)
                draw_points(img_bgr, target_pts_2d, color=(255, 0, 255), radius=2)

            # start
            T_start_cam = local_pose_to_camera_matrix(start_pose7d_local, centroid_0)
            start_origin = draw_3d_axes_on_image(
                img_bgr=img_bgr,
                T_camera=T_start_cam,
                intrinsics=intrinsics,
                centroid_ref=centroid_0,
                scale=AXIS_LENGTH,
                mode="start",
            )

            # trajectory line + sparse axes
            for i, T_cam in enumerate(pred_camera_matrices):
                origin = draw_3d_axes_on_image(
                    img_bgr=img_bgr,
                    T_camera=T_cam,
                    intrinsics=intrinsics,
                    centroid_ref=centroid_0,
                    scale=AXIS_LENGTH * 0.7,
                    mode="traj_sparse" if (i % DRAW_EVERY == 0 or i == len(pred_camera_matrices) - 1) else "silent",
                )

                pred_origins_2d.append(origin)

            draw_polyline(
                img_bgr,
                pred_origins_2d,
                color=(0, 165, 255),
                thickness=2,
            )

            # goal / endpoint
            T_goal_cam = local_pose_to_camera_matrix(goal_pose7d_local, centroid_0)
            goal_origin = draw_3d_axes_on_image(
                img_bgr=img_bgr,
                T_camera=T_goal_cam,
                intrinsics=intrinsics,
                centroid_ref=centroid_0,
                scale=AXIS_LENGTH * 1.1,
                mode="goal",
            )

            # endpoint 再画一次，强调 Full64 边界终点
            endpoint_origin = draw_3d_axes_on_image(
                img_bgr=img_bgr,
                T_camera=pred_camera_matrices[-1],
                intrinsics=intrinsics,
                centroid_ref=centroid_0,
                scale=AXIS_LENGTH * 1.0,
                mode="pred",
            )

            endpoint_err_cm = np.linalg.norm(
                pred_local_poses[-1, :3] - goal_pose7d_local[:3]
            ) * 100.0

            cv2.putText(
                img_bgr,
                f"Scene: {scene_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr,
                "Full64 Trajectory: orange line | sparse axes",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr,
                "Start: cyan | Goal: yellow/orange | Affordance: yellow | Target: magenta",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr,
                f"Endpoint boundary err: {endpoint_err_cm:.4f} cm",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ================= 保存 =================
            out_dir = os.path.join(scene_dir, OUTPUT_SUBDIR)
            ensure_dir(out_dir)

            overlay_path = os.path.join(out_dir, "pred_full64_trajectory_overlay.png")
            residual_path = os.path.join(out_dir, "pred_residual_actions.npy")
            local_pose_path = os.path.join(out_dir, "pred_local_poses.npy")
            camera_matrix_path = os.path.join(out_dir, "pred_camera_matrices.npy")
            origins_path = os.path.join(out_dir, "pred_origins_2d.npy")
            goal_debug_path = os.path.join(out_dir, "goal_condition_debug.npz")
            input_debug_path = os.path.join(out_dir, "input_reconstructed_pointclouds.npz")
            meta_path = os.path.join(out_dir, "inference_meta.json")

            cv2.imwrite(overlay_path, img_bgr)

            np.save(residual_path, pred_residual_actions)
            np.save(local_pose_path, pred_local_poses)
            np.save(camera_matrix_path, pred_camera_matrices)
            np.save(origins_path, np.asarray(pred_origins_2d, dtype=np.int32))

            np.savez_compressed(
                goal_debug_path,
                first_stage_goal_pose7d=first_stage_goal_pose7d,
                goal_pose7d_local=goal_pose7d_local,
                goal_pose9d=debug_goal["goal_pose9d"],
                goal_delta_pose7d=debug_goal["goal_delta_pose7d"],
                goal_delta_pose9d=debug_goal["goal_delta_pose9d"],
                T_start=debug_goal["T_start"],
                T_goal=debug_goal["T_goal"],
                T_delta=debug_goal["T_delta"],
                start_origin_2d=start_origin,
                goal_origin_2d=goal_origin,
                endpoint_origin_2d=endpoint_origin,
            )

            np.savez_compressed(
                input_debug_path,
                pc_man_0=pc_man_0,
                pc_tgt_0=pc_tgt_0,
                pc_man_0_local=pc_man_0_local,
                pc_tgt_0_local=pc_tgt_0_local,
                centroid_0=centroid_0,
                affordance_pts_2d=affordance_pts_2d,
                target_pts_2d=target_pts_2d,
            )

            meta = {
                "scene": scene_name,
                "scene_dir": scene_dir,
                "second_stage_checkpoint": SECOND_STAGE_CKPT_PATH,
                "first_stage_goal_pose_path": first_goal_path,
                "first_stage_goal_mode": FIRST_STAGE_GOAL_MODE,
                "output_dir": out_dir,
                "overlay_path": overlay_path,
                "num_pts": int(num_pts),
                "num_pred_steps": int(pred_residual_actions.shape[0]),
                "draw_every": int(DRAW_EVERY),
                "intrinsics_source": INTRINSICS_SOURCE,
                "depth_scale": float(depth_scale),
                "affordance_points_path": affordance_pts_path,
                "target_points_path": target_pts_path,
                "lang_emb_source": lang_emb_source,
                "centroid_0": centroid_0.tolist(),
                "goal_pose7d_local": goal_pose7d_local.tolist(),
                "goal_delta_pose7d": debug_goal["goal_delta_pose7d"].tolist(),
                "endpoint_boundary_err_cm": float(endpoint_err_cm),
                "saved_files": {
                    "overlay": overlay_path,
                    "pred_residual_actions": residual_path,
                    "pred_local_poses": local_pose_path,
                    "pred_camera_matrices": camera_matrix_path,
                    "pred_origins_2d": origins_path,
                    "goal_condition_debug": goal_debug_path,
                    "input_reconstructed_pointclouds": input_debug_path,
                },
                "note": (
                    "This script reads only the first-stage pred_goal_pose7d.npy as terminal goal. "
                    "Point clouds are reconstructed again from scene affordance/target sample points and depth. "
                    "The second-stage Full64 policy generates residual trajectory with start/end boundary inpainting."
                ),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            summary_rows.append(
                {
                    "scene": scene_name,
                    "status": "ok",
                    "overlay_path": overlay_path,
                    "out_dir": out_dir,
                    "endpoint_boundary_err_cm": float(endpoint_err_cm),
                    "goal_x": float(goal_pose7d_local[0]),
                    "goal_y": float(goal_pose7d_local[1]),
                    "goal_z": float(goal_pose7d_local[2]),
                    "first_goal_path": first_goal_path,
                }
            )

            print(f"[✔] 保存可视化: {overlay_path}")
            print(f"[✔] 保存轨迹结果: {out_dir}")

        except Exception as exc:
            print(f"[✘] {scene_name} 推理失败: {exc}")
            summary_rows.append(
                {
                    "scene": scene_name,
                    "status": "failed",
                    "error": str(exc),
                    "overlay_path": "",
                    "out_dir": "",
                    "endpoint_boundary_err_cm": "",
                    "goal_x": "",
                    "goal_y": "",
                    "goal_z": "",
                    "first_goal_path": "",
                }
            )

    summary_json_path = os.path.join(SCENE_ROOT, "full64_trajectory_inference_summary.json")
    summary_csv_path = os.path.join(SCENE_ROOT, "full64_trajectory_inference_summary.csv")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    if len(summary_rows) > 0:
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n[*] summary json saved to: {summary_json_path}")
    print(f"[*] summary csv saved to: {summary_csv_path}")
    print("🎉 第二阶段 Full64 轨迹推理完成。")


if __name__ == "__main__":
    main()
