import os
import glob
import sys
import numpy as np
import torch
import json
import zarr
from typing import Dict
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from diffusion_policy_3d.model.goal.pose_utils import pose7d_to_pose9d

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


# ================= 工具函数 =================

def install_numpy_pickle_compat():
    """
    NumPy 2.x pickles may reference numpy._core.*.
    The spot env runs Python 3.8 with NumPy 1.24.x, where those modules live
    under numpy.core.*. Register aliases before np.load(..., allow_pickle=True).
    """
    aliases = {
        "numpy._core": np.core,
        "numpy._core.multiarray": np.core.multiarray,
        "numpy._core.numeric": np.core.numeric,
        "numpy._core.umath": np.core.umath,
        "numpy._core._multiarray_umath": np.core._multiarray_umath,
    }
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


install_numpy_pickle_compat()

def load_task_lang_emb(task_dir: str, use_lang_emb: bool, missing_lang_emb: str = "zero"):
    if not use_lang_emb:
        return None

    lang_path = os.path.join(task_dir, "lang_emb.npy")
    if os.path.exists(lang_path):
        lang_emb = np.load(lang_path).astype(np.float32)
        if lang_emb.ndim == 1:
            lang_emb = lang_emb[None, :]
        return lang_emb

    if missing_lang_emb == "zero":
        print(f"[LFVDataset] missing {lang_path}; using zero language embedding [1,1024].")
        return np.zeros((1, 1024), dtype=np.float32)
    if missing_lang_emb == "error":
        raise FileNotFoundError(
            f"Missing language embedding: {lang_path}. "
            "Run `python -m diffusion_policy_3d.dataset.generate_lang_emb`, "
            "set dataset.use_lang_emb=false, or set dataset.missing_lang_emb=zero."
        )
    if missing_lang_emb in ("none", None):
        return None
    raise ValueError(f"Unsupported missing_lang_emb policy: {missing_lang_emb!r}")

def intrinsics_dict_to_matrix(intrinsics_raw):
    fx = intrinsics_raw.get("fx", intrinsics_raw.get("f"))
    fy = intrinsics_raw.get("fy", intrinsics_raw.get("f"))
    cx = intrinsics_raw.get("cx", intrinsics_raw.get("ppx"))
    cy = intrinsics_raw.get("cy", intrinsics_raw.get("ppy"))
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(f"Invalid intrinsics dictionary: {intrinsics_raw}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def load_episode_camera_params(ep_path, intrinsics_source="depth_intrinsics_original"):
    meta_path = os.path.join(ep_path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing episode meta.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if intrinsics_source not in meta:
        raise KeyError(f"{meta_path} does not contain {intrinsics_source!r}")
    intrinsics = intrinsics_dict_to_matrix(meta[intrinsics_source])
    depth_scale = float(meta.get("depth_scale", 1.0))
    return intrinsics, depth_scale


def unproject_2d_to_3d(pts_2d, depth_map, intrinsics, target_num=64):
    """将 2D 像素点根据深度图反投影为 3D 点云"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    pts_3d = []
    for p in pts_2d:
        x, y = int(p[0]), int(p[1])
        if y >= depth_map.shape[0] or x >= depth_map.shape[1]:
            continue

        z_c = float(depth_map[y, x])
        if z_c <= 0 or np.isnan(z_c):
            continue

        x_c = (x - cx) * z_c / fx
        y_c = (y - cy) * z_c / fy
        pts_3d.append([x_c, y_c, z_c])

    pts_3d = np.array(pts_3d, dtype=np.float32)

    # 填充/降采样到固定点数
    if len(pts_3d) == 0:
        return np.zeros((target_num, 3), dtype=np.float32)
    elif len(pts_3d) > target_num:
        indices = np.random.choice(len(pts_3d), target_num, replace=False)
        pts_3d = pts_3d[indices]
    elif len(pts_3d) < target_num:
        pad_indices = np.random.choice(len(pts_3d), target_num - len(pts_3d), replace=True)
        pts_3d = np.vstack([pts_3d, pts_3d[pad_indices]])

    return pts_3d


def pose_to_matrix(pose):
    """将 7D 位姿 [x, y, z, qx, qy, qz, qw] 转为 4x4 变换矩阵"""
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return T


def matrix_to_pose(T):
    """将 4x4 变换矩阵转回 7D 位姿"""
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = T[:3, 3]
    pose[3:7] = R.from_matrix(T[:3, :3]).as_quat()
    return pose


# def sequence_to_residual(sequence_abs, current_idx, zero_prefix=True):
#     """
#     将窗口内的“绝对局部位姿序列”转换为“相对于当前时刻”的残差序列

#     输入:
#         sequence_abs: [T, D]
#             前7维为 [x, y, z, qx, qy, qz, qw]
#             后续维度(如progress/gripper)保持原值，不做SE(3)残差化
#         current_idx: 当前时刻在窗口中的索引，通常是 obs_horizon - 1
#         zero_prefix: 是否将 current_idx 之前的前缀强制置为单位残差
#                      这样训练更聚焦“当前到未来”的预测

#     输出:
#         sequence_rel: [T, D]
#             前7维为 T_curr^{-1} * T_i
#     """
#     sequence_rel = np.zeros_like(sequence_abs, dtype=np.float32)

#     T_curr = pose_to_matrix(sequence_abs[current_idx, :7])
#     T_curr_inv = np.linalg.inv(T_curr)

#     for i in range(sequence_abs.shape[0]):
#         T_i = pose_to_matrix(sequence_abs[i, :7])
#         T_rel = T_curr_inv @ T_i
#         sequence_rel[i, :7] = matrix_to_pose(T_rel)

#         if sequence_abs.shape[1] > 7:
#             sequence_rel[i, 7:] = sequence_abs[i, 7:]

#     # 四元数连续性
#     for i in range(1, sequence_rel.shape[0]):
#         if np.dot(sequence_rel[i, 3:7], sequence_rel[i - 1, 3:7]) < 0:
#             sequence_rel[i, 3:7] = -sequence_rel[i, 3:7]

#     if zero_prefix and current_idx > 0:
#         identity_pose = np.zeros(sequence_abs.shape[1], dtype=np.float32)
#         identity_pose[6] = 1.0  # qw = 1
#         if sequence_abs.shape[1] > 7:
#             identity_pose[7:] = sequence_rel[current_idx, 7:]
#         sequence_rel[:current_idx] = identity_pose

#     return sequence_rel

def sequence_to_residual(sequence_abs, current_idx, zero_prefix=True):
    """
    7D absolute local pose sequence -> 7D residual pose sequence
    将窗口内的“绝对局部位姿序列”转换为“相对于当前时刻”的残差序列
    输入:
        sequence_abs: [T, 7]
        current_idx: 当前时刻索引
    输出:
        sequence_rel: [T, 7]
    """
    sequence_rel = np.zeros_like(sequence_abs, dtype=np.float32)

    T_curr = pose_to_matrix(sequence_abs[current_idx, :7])
    T_curr_inv = np.linalg.inv(T_curr)

    for i in range(sequence_abs.shape[0]):
        T_i = pose_to_matrix(sequence_abs[i, :7])
        T_rel = T_curr_inv @ T_i
        sequence_rel[i, :7] = matrix_to_pose(T_rel)

    # 四元数连续
    for i in range(1, sequence_rel.shape[0]):
        if np.dot(sequence_rel[i, 3:7], sequence_rel[i - 1, 3:7]) < 0:
            sequence_rel[i, 3:7] = -sequence_rel[i, 3:7]

    if zero_prefix and current_idx > 0:
        identity_pose = np.zeros(7, dtype=np.float32)
        identity_pose[6] = 1.0
        sequence_rel[:current_idx] = identity_pose

    return sequence_rel


# ================= 共享滑窗 episode 底座 =================

class LFVSlidingWindowSE3Dataset(BaseDataset):
    def __init__(
        self,
        data_dirs: list,
        pred_horizon=16,
        obs_horizon=1,
        action_horizon=8,
        use_lang_emb=True,
        mode='train',
        val_ratio=0.2,
        intrinsics_source="depth_intrinsics_original",
        missing_lang_emb="zero",
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_lang_emb = use_lang_emb
        self.mode = mode
        self.val_ratio = val_ratio
        self.missing_lang_emb = missing_lang_emb

        self.num_pts = 64  # 点云采样数量

        self.intrinsics_source = intrinsics_source
        self.intrinsics = None

        self.trajectories = []
        self.pc_man_0_list = []
        self.pc_tgt_0_list = []
        self.traj_lang_embs = []
        self.indices = []

        self._load_data()
        print(f"[Dataset - {self.mode.upper()}] 加载完成！总滑动窗口数: {len(self.indices)}")

    def _load_data(self):
        for task_dir in self.data_dirs:
            # 1. 语言特征
            task_lang_emb = load_task_lang_emb(task_dir, self.use_lang_emb, self.missing_lang_emb)

            # 2. 数据切分
            all_episodes = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
            np.random.seed(42)
            np.random.shuffle(all_episodes)
            split_idx = int(len(all_episodes) * (1 - self.val_ratio))
            selected_episodes = all_episodes[:split_idx] if self.mode == 'train' else all_episodes[split_idx:]

            for ep_path in selected_episodes:
                npz_path = os.path.join(ep_path, "se3_trajectory", "dp_action_trajectory.npz")
                if not os.path.exists(npz_path):
                    continue

                # 剪裁轨迹去掉第八个维度
                traj = np.load(npz_path)['actions_8d'].astype(np.float32)[:, :7]
                T_frames = traj.shape[0]

                depth_path = os.path.join(ep_path, "depth")
                depth_0 = None
                intrinsics = None
                if os.path.exists(depth_path):
                    try:
                        intrinsics, depth_scale = load_episode_camera_params(ep_path, self.intrinsics_source)
                        depth_zarr = zarr.open(depth_path, mode='r')
                        depth_0 = depth_zarr[0].astype(np.float32) * depth_scale
                        self.intrinsics = intrinsics
                    except Exception as exc:
                        print(f"[LFVSlidingWindowSE3Dataset] failed camera params/depth for {ep_path}: {exc}")

                # manipulated 第一帧点云
                man_2d_path = os.path.join(ep_path, "sample_points", "sampled_2d_uniform.npy")
                if os.path.exists(man_2d_path) and depth_0 is not None and intrinsics is not None:
                    pts_2d_man = np.load(man_2d_path, allow_pickle=True).item()['query_points_2d']
                    pc_man_0 = unproject_2d_to_3d(pts_2d_man, depth_0, intrinsics, self.num_pts)
                else:
                    pc_man_0 = np.zeros((self.num_pts, 3), dtype=np.float32)

                # target 第一帧点云
                tgt_2d_path = os.path.join(ep_path, "target_sample_points", "target_sampled_2d_uniform.npy")
                if os.path.exists(tgt_2d_path) and depth_0 is not None and intrinsics is not None:
                    pts_2d_tgt = np.load(tgt_2d_path, allow_pickle=True).item()['query_points_2d']
                    pc_tgt_0 = unproject_2d_to_3d(pts_2d_tgt, depth_0, intrinsics, self.num_pts)
                else:
                    pc_tgt_0 = np.zeros((self.num_pts, 3), dtype=np.float32)

                self.trajectories.append(traj)
                self.pc_man_0_list.append(pc_man_0)
                self.pc_tgt_0_list.append(pc_tgt_0)
                self.traj_lang_embs.append(task_lang_emb)

                traj_idx = len(self.trajectories) - 1
                for t in range(T_frames):
                    self.indices.append((traj_idx, t))

    def get_validation_dataset(self):
        return LFVSlidingWindowSE3Dataset(
            data_dirs=self.data_dirs,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            use_lang_emb=self.use_lang_emb,
            mode='val',
            val_ratio=self.val_ratio,
            intrinsics_source=self.intrinsics_source,
            missing_lang_emb=self.missing_lang_emb,
        )

    def get_normalizer(self, mode='limits', **kwargs):
        """
        action: 归一化 residual action 的平移部分
        agent_pos: 归一化 observation 的绝对局部平移部分
        """
        action_all = []
        agent_pos_all = []

        for traj, pc_man_0 in zip(self.trajectories, self.pc_man_0_list):
            centroid_0 = np.mean(pc_man_0, axis=0)

            # 整条轨迹转到“绝对局部位姿”
            local_traj = traj.copy()[:, :7]
            for i in range(local_traj.shape[0]):
                R_seq = R.from_quat(local_traj[i, 3:7]).as_matrix()
                t_seq = local_traj[i, :3]
                local_traj[i, :3] = R_seq @ centroid_0 + t_seq - centroid_0

            # 四元数连续
            for i in range(1, local_traj.shape[0]):
                if np.dot(local_traj[i, 3:7], local_traj[i - 1, 3:7]) < 0:
                    local_traj[i, 3:7] = -local_traj[i, 3:7]

            # agent_pos 统计：绝对局部位置
            agent_pos_all.append(local_traj[:, :3])

            # action 统计：窗口中的 residual 平移
            T_frames = local_traj.shape[0]
            for t in range(T_frames):
                start_idx = t - self.obs_horizon + 1
                end_idx = start_idx + self.pred_horizon
                window_indices = np.clip(np.arange(start_idx, end_idx), 0, T_frames - 1)

                seq_abs = local_traj[window_indices].copy()
                current_idx = self.obs_horizon - 1
                seq_rel = sequence_to_residual(seq_abs, current_idx, zero_prefix=True)

                # 只统计从当前时刻开始的 residual，更符合执行语义
                action_all.append(seq_rel[current_idx:, :3])

        action_all = np.concatenate(action_all, axis=0)
        agent_pos_all = np.concatenate(agent_pos_all, axis=0)

        data = {
            'action': action_all,
            'agent_pos': agent_pos_all,
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.indices)

    def _get_se3_augmentation(self):
        """生成随机微小 SE(3) 扰动矩阵"""
        euler_noise = np.random.uniform(-5, 5, size=3)
        R_aug = R.from_euler('xyz', euler_noise, degrees=True).as_matrix()
        t_aug = np.random.uniform(-0.02, 0.02, size=3)

        T_aug = np.eye(4, dtype=np.float32)
        T_aug[:3, :3] = R_aug
        T_aug[:3, 3] = t_aug
        return T_aug

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, t = self.indices[idx]
        traj = self.trajectories[traj_idx]
        pc_man_0 = self.pc_man_0_list[traj_idx]
        pc_tgt_0 = self.pc_tgt_0_list[traj_idx]

        lang_emb = None
        if self.use_lang_emb and len(self.traj_lang_embs) > 0:
            lang_emb = self.traj_lang_embs[traj_idx]

        # 窗口切片
        start_idx = t - self.obs_horizon + 1
        end_idx = start_idx + self.pred_horizon
        window_indices = np.clip(np.arange(start_idx, end_idx), 0, traj.shape[0] - 1)

        # sequence: 原始“相对第一帧”的窗口序列
        sequence = traj[window_indices].copy()[:, :7]

        # ====================================================================
        # 1️⃣ 当前点云：用当前时刻位姿推演 manipulated 点云
        # ====================================================================
        current_pose = sequence[self.obs_horizon - 1]
        T_0_to_t = pose_to_matrix(current_pose[:7])

        pc_man_0_h = np.concatenate([pc_man_0, np.ones((self.num_pts, 1))], axis=1)
        curr_pc_man = (T_0_to_t @ pc_man_0_h.T).T[:, :3].astype(np.float32)
        curr_pc_tgt = pc_tgt_0.copy()

        # ====================================================================
        # 2️⃣ 视觉局部化：统一减去第一帧 manipulated 质心
        # ====================================================================
        centroid_0 = np.mean(pc_man_0, axis=0)
        curr_pc_man = curr_pc_man - centroid_0
        curr_pc_tgt = curr_pc_tgt - centroid_0

        # ====================================================================
        # 3️⃣ 把窗口内的轨迹转为“绝对局部位姿”
        # ====================================================================
        for i in range(sequence.shape[0]):
            R_seq = R.from_quat(sequence[i, 3:7]).as_matrix()
            t_seq = sequence[i, :3]
            sequence[i, :3] = R_seq @ centroid_0 + t_seq - centroid_0

        # ====================================================================
        # 4️⃣ SE(3) 增强：同时增强点云与“绝对局部位姿”
        # ====================================================================
        if self.mode == 'train':
            T_aug = self._get_se3_augmentation()
            T_aug_inv = np.linalg.inv(T_aug)

            curr_pc_man_h = np.concatenate([curr_pc_man, np.ones((self.num_pts, 1))], axis=1)
            curr_pc_tgt_h = np.concatenate([curr_pc_tgt, np.ones((self.num_pts, 1))], axis=1)
            curr_pc_man = (T_aug @ curr_pc_man_h.T).T[:, :3].astype(np.float32)
            curr_pc_tgt = (T_aug @ curr_pc_tgt_h.T).T[:, :3].astype(np.float32)

            for i in range(sequence.shape[0]):
                T_seq_local = pose_to_matrix(sequence[i, :7])
                T_seq_aug = T_aug @ T_seq_local @ T_aug_inv
                sequence[i, :7] = matrix_to_pose(T_seq_aug)

        # ====================================================================
        # 5️⃣ 四元数连续性修复
        # ====================================================================
        for i in range(1, sequence.shape[0]):
            if np.dot(sequence[i, 3:7], sequence[i - 1, 3:7]) < 0:
                sequence[i, 3:7] = -sequence[i, 3:7]

        # ====================================================================
        # 6️⃣ obs = 当前绝对局部位姿
        #    action = 相对于当前时刻的 residual trajectory
        # ====================================================================
        obs_seq = sequence[:self.obs_horizon, :].copy()
        current_idx = self.obs_horizon - 1
        action_seq = sequence_to_residual(sequence, current_idx, zero_prefix=True)

        # 点云时间维对齐
        pc_man_tensor = torch.from_numpy(curr_pc_man).float().unsqueeze(0).repeat(self.obs_horizon, 1, 1)
        pc_tgt_tensor = torch.from_numpy(curr_pc_tgt).float().unsqueeze(0).repeat(self.obs_horizon, 1, 1)

        data_dict = {
            "obs": {
                "agent_pos": torch.from_numpy(obs_seq).float(),
                "pc_manipulated": pc_man_tensor,
                "pc_target": pc_tgt_tensor
            },
            "action": torch.from_numpy(action_seq).float()
        }

        if self.use_lang_emb and lang_emb is not None:
            data_dict["obs"]["lang_token_embs"] = torch.from_numpy(lang_emb).float()

        return data_dict


# ================= 第一阶段 goal pose 数据集 =================

def _resize_points(points: np.ndarray, num_pts: int) -> np.ndarray:
    """
    与原 GoalPoseSE3Dataset 保持一致：
    - 点数正好：直接返回
    - 点数过多：用 linspace 做确定性下采样
    - 点数不足：循环补齐
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


def raw_pose7d_to_local_pose7d(raw_pose7d: np.ndarray, centroid_0: np.ndarray) -> np.ndarray:
    """
    将 dp_action_trajectory.npz 中的 raw camera-space pose 转成
    以第一帧 manipulated object centroid 为原点的 local pose。

    raw pose 定义：
        p_cam' = R @ p_cam + t_raw

    local point 定义：
        p_local = p_cam - C0
        p_local' = p_cam' - C0

    因此：
        p_local' = R @ p_local + t_local
        t_local = R @ C0 + t_raw - C0

    输入:
        raw_pose7d: [x, y, z, qx, qy, qz, qw]
        centroid_0: [3]

    输出:
        local_pose7d: [x_local, y_local, z_local, qx, qy, qz, qw]
    """
    raw_pose7d = np.asarray(raw_pose7d, dtype=np.float32).copy()
    centroid_0 = np.asarray(centroid_0, dtype=np.float32).reshape(3)

    if raw_pose7d.shape[-1] < 7:
        raise ValueError(f"raw_pose7d expected at least 7 dims, got shape {raw_pose7d.shape}")

    quat = raw_pose7d[3:7].astype(np.float32)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        raise ValueError(f"Invalid quaternion with near-zero norm: {quat}")

    quat = quat / quat_norm
    R_raw = R.from_quat(quat).as_matrix().astype(np.float32)
    t_raw = raw_pose7d[:3].astype(np.float32)

    local_pose7d = raw_pose7d[:7].copy()
    local_pose7d[:3] = R_raw @ centroid_0 + t_raw - centroid_0
    local_pose7d[3:7] = quat
    return local_pose7d.astype(np.float32)


class GoalPoseSE3Dataset(BaseDataset):
    """
    第一阶段 GoalPoseDiffuser 数据集。

    当前语义：
    - 输入点云:
        pc_manipulated = pc_man_0 - centroid_0
        pc_target      = pc_tgt_0 - centroid_0

    - 输出标签:
        goal_pose7d 是 terminal raw pose 转换后的 centroid-local pose。

    这样模型学习的是：
        local manipulated cloud + local target cloud -> local terminal pose
    """

    def __init__(
        self,
        data_dirs: list,
        pred_horizon=16,
        obs_horizon=1,
        action_horizon=8,
        use_lang_emb=True,
        num_pts=256,
        mode="train",
        val_ratio=0.1,
        intrinsics_source="depth_intrinsics_original",
        missing_lang_emb="zero",
    ):
        super().__init__()

        self.data_dirs = data_dirs
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_lang_emb = use_lang_emb
        self.num_pts = num_pts
        self.mode = mode
        self.val_ratio = val_ratio
        self.intrinsics_source = intrinsics_source
        self.missing_lang_emb = missing_lang_emb

        self.base_dataset = LFVSlidingWindowSE3Dataset(
            data_dirs=data_dirs,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            use_lang_emb=use_lang_emb,
            mode=mode,
            val_ratio=val_ratio,
            intrinsics_source=intrinsics_source,
            missing_lang_emb=missing_lang_emb,
        )

        self.trajectories = self.base_dataset.trajectories
        self.pc_man_0_list = self.base_dataset.pc_man_0_list
        self.pc_tgt_0_list = self.base_dataset.pc_tgt_0_list
        self.traj_lang_embs = self.base_dataset.traj_lang_embs
        self.intrinsics = getattr(self.base_dataset, "intrinsics", None)
        self.intrinsics_source = getattr(self.base_dataset, "intrinsics_source", intrinsics_source)

        # GoalPoseSE3Dataset 以 episode 为样本单位，不再使用 sliding window。
        self.indices = list(range(len(self.trajectories)))

        # 仅用于可视化和 debug，不参与训练。
        self.episode_paths = self._infer_episode_paths()

    def _infer_episode_paths(self):
        episode_paths = []

        for task_dir in self.data_dirs:
            all_episodes = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
            rng = np.random.RandomState(42)
            rng.shuffle(all_episodes)

            split_idx = int(len(all_episodes) * (1 - self.val_ratio))
            selected = all_episodes[:split_idx] if self.mode == "train" else all_episodes[split_idx:]

            for ep_path in selected:
                npz_path = os.path.join(ep_path, "se3_trajectory", "dp_action_trajectory.npz")
                if os.path.exists(npz_path):
                    episode_paths.append(ep_path)

        if len(episode_paths) != len(self.trajectories):
            print(
                f"[GoalPoseSE3Dataset] Warning: inferred episode_paths length "
                f"{len(episode_paths)} != trajectories length {len(self.trajectories)}. "
                f"Visualization path alignment may be unreliable."
            )
            episode_paths = episode_paths[: len(self.trajectories)]

        return episode_paths

    def __len__(self) -> int:
        return len(self.indices)

    def _get_episode_local_data(self, traj_idx: int):
        """
        统一处理单条 episode 的：
        - resized raw point clouds
        - centroid_0
        - local point clouds
        - raw terminal pose
        - local terminal pose

        __getitem__ 和 get_normalizer 都调用这里，避免两处逻辑不一致。
        """
        traj = self.trajectories[traj_idx]

        pc_man_0 = _resize_points(self.pc_man_0_list[traj_idx], self.num_pts)
        pc_tgt_0 = _resize_points(self.pc_tgt_0_list[traj_idx], self.num_pts)

        centroid_0 = pc_man_0.mean(axis=0).astype(np.float32)

        pc_man_local = (pc_man_0 - centroid_0).astype(np.float32)
        pc_tgt_local = (pc_tgt_0 - centroid_0).astype(np.float32)

        raw_goal_pose7d = traj[-1, :7].astype(np.float32).copy()
        local_goal_pose7d = raw_pose7d_to_local_pose7d(raw_goal_pose7d, centroid_0)

        return {
            "pc_man_0": pc_man_0,
            "pc_tgt_0": pc_tgt_0,
            "centroid_0": centroid_0,
            "pc_man_local": pc_man_local,
            "pc_tgt_local": pc_tgt_local,
            "raw_goal_pose7d": raw_goal_pose7d,
            "local_goal_pose7d": local_goal_pose7d,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = self.indices[idx]
        item = self._get_episode_local_data(traj_idx)

        goal_pose7d = item["local_goal_pose7d"]
        goal_pose9d = pose7d_to_pose9d(
            torch.from_numpy(goal_pose7d).float()
        ).numpy().astype(np.float32)

        data = {
            "obs": {
                "pc_manipulated": torch.from_numpy(item["pc_man_local"]).float(),
                "pc_target": torch.from_numpy(item["pc_tgt_local"]).float(),
                "agent_pos": torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32),
            },
            # 训练真正使用的标签：local terminal pose
            "goal_pose7d": torch.from_numpy(goal_pose7d).float(),
            "goal_pose9d": torch.from_numpy(goal_pose9d).float(),

            # debug / visualization 辅助字段
            "raw_goal_pose7d": torch.from_numpy(item["raw_goal_pose7d"]).float(),
            "centroid_0": torch.from_numpy(item["centroid_0"]).float(),

            "traj_idx": torch.tensor(traj_idx, dtype=torch.long),
        }

        if (
            self.use_lang_emb
            and len(self.traj_lang_embs) > 0
            and self.traj_lang_embs[traj_idx] is not None
        ):
            data["obs"]["lang_token_embs"] = torch.from_numpy(
                self.traj_lang_embs[traj_idx]
            ).float()

        return data

    def get_validation_dataset(self):
        return GoalPoseSE3Dataset(
            data_dirs=self.data_dirs,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            use_lang_emb=self.use_lang_emb,
            num_pts=self.num_pts,
            mode="val",
            val_ratio=self.val_ratio,
            intrinsics_source=self.intrinsics_source,
            missing_lang_emb=self.missing_lang_emb,
        )

    def get_normalizer(self, mode="limits", **kwargs):
        """
        必须使用和 __getitem__ 完全一致的 local goal pose 统计 normalizer。

        之前的问题是：
            __getitem__ 输入点云是 local，
            但 get_normalizer 和 goal label 都是 raw pose。

        修改后：
            normalizer 统计 local goal_pose9d。
        """
        goal_pose9d = []

        for traj_idx in range(len(self.trajectories)):
            item = self._get_episode_local_data(traj_idx)
            local_goal_pose7d = item["local_goal_pose7d"]

            local_goal_pose9d = pose7d_to_pose9d(
                torch.from_numpy(local_goal_pose7d).float()
            ).numpy().astype(np.float32)

            goal_pose9d.append(local_goal_pose9d)

        if len(goal_pose9d) == 0:
            raise RuntimeError(
                "GoalPoseSE3Dataset is empty; check configs/model/task/goal_pose_multitask.yaml data_dirs."
            )

        normalizer = LinearNormalizer()
        normalizer.fit(
            {"goal_pose9d": np.stack(goal_pose9d).astype(np.float32)},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer


# ================= 第二阶段 full64 轨迹数据集工具 =================

def pose7d_to_matrix_np(pose: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    quat = np.asarray(pose[3:7], dtype=np.float32)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    T[:3, :3] = R.from_quat(quat).as_matrix().astype(np.float32)
    T[:3, 3] = np.asarray(pose[:3], dtype=np.float32)
    return T


def matrix_to_pose7d_np(T: np.ndarray) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = T[:3, 3]
    pose[3:7] = R.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
    return pose


def matrix_to_pose9d_np(T: np.ndarray) -> np.ndarray:
    pose = np.zeros(9, dtype=np.float32)
    pose[:3] = T[:3, 3]
    pose[3:9] = T[:3, :2].T.reshape(6).astype(np.float32)
    return pose


def ensure_quat_continuity(traj: np.ndarray) -> np.ndarray:
    traj = traj.copy()
    for i in range(1, traj.shape[0]):
        if np.dot(traj[i, 3:7], traj[i - 1, 3:7]) < 0:
            traj[i, 3:7] = -traj[i, 3:7]
    return traj.astype(np.float32)


def resample_se3_trajectory(traj: np.ndarray, horizon: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape[0] == horizon:
        return ensure_quat_continuity(traj)
    if traj.shape[0] < 2:
        return np.repeat(traj[:1], horizon, axis=0).astype(np.float32)

    src_t = np.linspace(0.0, 1.0, traj.shape[0])
    dst_t = np.linspace(0.0, 1.0, horizon)
    trans = np.stack(
        [np.interp(dst_t, src_t, traj[:, i]) for i in range(3)],
        axis=-1
    ).astype(np.float32)

    traj = ensure_quat_continuity(traj)
    rots = R.from_quat(traj[:, 3:7])
    slerp = Slerp(src_t, rots)
    quat = slerp(dst_t).as_quat().astype(np.float32)

    out = np.concatenate([trans, quat], axis=-1).astype(np.float32)
    return ensure_quat_continuity(out)


def resize_points(points: np.ndarray, num_pts: int) -> np.ndarray:
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


class FullTrajectoryGoalConditionedSE3Dataset(BaseDataset):
    """
    Episode-level full64 dataset for goal-conditioned trajectory diffusion.

    Invariants preserved:
    - raw pose: [x, y, z, qx, qy, qz, qw], quaternion xyzw
    - pc local: pc - first-frame manipulated centroid
    - t_local = R @ centroid_0 + t_raw - centroid_0
    - action[i] = inv(T_start) @ T_i, not step-to-step increments
    """

    def __init__(
        self,
        data_dirs: list,
        horizon=64,
        obs_horizon=1,
        action_horizon=64,
        use_lang_emb=True,
        num_pts=64,
        mode="train",
        val_ratio=0.1,
        resample_to_horizon=True,
        use_se3_aug=True,
        intrinsics_source="depth_intrinsics_original",
        missing_lang_emb="zero",
    ):
        super().__init__()
        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        else:
            self.data_dirs = [str(p) for p in list(data_dirs)]
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_lang_emb = use_lang_emb
        self.num_pts = num_pts
        self.mode = mode
        self.val_ratio = val_ratio
        self.resample_to_horizon = resample_to_horizon
        self.use_se3_aug = use_se3_aug
        self.intrinsics_source = intrinsics_source
        self.missing_lang_emb = missing_lang_emb
        self.intrinsics = None

        self.trajectories = []
        self.pc_man_0_list = []
        self.pc_tgt_0_list = []
        self.traj_lang_embs = []
        self.episode_paths = []
        self._load_data()
        print(f"[Full64GoalDataset - {self.mode.upper()}] episodes: {len(self.trajectories)}")

    def _load_data(self):
        rng = np.random.RandomState(42)
        for task_dir in self.data_dirs:
            task_lang_emb = load_task_lang_emb(task_dir, self.use_lang_emb, self.missing_lang_emb)

            all_episodes = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
            rng.shuffle(all_episodes)
            split_idx = int(len(all_episodes) * (1 - self.val_ratio))
            selected = all_episodes[:split_idx] if self.mode == "train" else all_episodes[split_idx:]

            for ep_path in selected:
                npz_path = os.path.join(ep_path, "se3_trajectory", "dp_action_trajectory.npz")
                if not os.path.exists(npz_path):
                    continue
                traj = np.load(npz_path)["actions_8d"].astype(np.float32)[:, :7]
                if traj.shape[0] == 0:
                    continue

                pc_man_0, pc_tgt_0 = self._load_episode_point_clouds(ep_path)
                self.trajectories.append(traj)
                self.pc_man_0_list.append(pc_man_0)
                self.pc_tgt_0_list.append(pc_tgt_0)
                self.traj_lang_embs.append(task_lang_emb)
                self.episode_paths.append(ep_path)

    def _load_episode_point_clouds(self, ep_path: str):
        depth_path = os.path.join(ep_path, "depth")
        depth_zarr = None
        depth_0 = None
        intrinsics = None
        if os.path.exists(depth_path):
            try:
                intrinsics, depth_scale = load_episode_camera_params(ep_path, self.intrinsics_source)
                import zarr
                depth_zarr = zarr.open(depth_path, mode="r")
                depth_0 = depth_zarr[0].astype(np.float32) * depth_scale
                self.intrinsics = intrinsics
            except Exception as exc:
                print(f"[Full64GoalDataset] failed camera params/depth for {ep_path}: {exc}")
                depth_zarr = None

        pc_man_0 = np.zeros((self.num_pts, 3), dtype=np.float32)
        pc_tgt_0 = np.zeros((self.num_pts, 3), dtype=np.float32)

        man_2d_path = os.path.join(ep_path, "sample_points", "sampled_2d_uniform.npy")
        if intrinsics is not None and depth_0 is not None and os.path.exists(man_2d_path):
            pts_2d = np.load(man_2d_path, allow_pickle=True).item()["query_points_2d"]
            pc_man_0 = unproject_2d_to_3d(pts_2d, depth_0, intrinsics, self.num_pts)

        tgt_2d_path = os.path.join(ep_path, "target_sample_points", "target_sampled_2d_uniform.npy")
        if intrinsics is not None and depth_0 is not None and os.path.exists(tgt_2d_path):
            pts_2d = np.load(tgt_2d_path, allow_pickle=True).item()["query_points_2d"]
            pc_tgt_0 = unproject_2d_to_3d(pts_2d, depth_0, intrinsics, self.num_pts)

        return resize_points(pc_man_0, self.num_pts), resize_points(pc_tgt_0, self.num_pts)

    def __len__(self) -> int:
        return len(self.trajectories)

    def get_validation_dataset(self):
        return FullTrajectoryGoalConditionedSE3Dataset(
            data_dirs=self.data_dirs,
            horizon=self.horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            use_lang_emb=self.use_lang_emb,
            num_pts=self.num_pts,
            mode="val",
            val_ratio=self.val_ratio,
            resample_to_horizon=self.resample_to_horizon,
            use_se3_aug=self.use_se3_aug,
            intrinsics_source=self.intrinsics_source,
            missing_lang_emb=self.missing_lang_emb,
        )

    def _get_se3_augmentation(self):
        euler_noise = np.random.uniform(-5, 5, size=3)
        R_aug = R.from_euler("xyz", euler_noise, degrees=True).as_matrix()
        t_aug = np.random.uniform(-0.02, 0.02, size=3)
        T_aug = np.eye(4, dtype=np.float32)
        T_aug[:3, :3] = R_aug
        T_aug[:3, 3] = t_aug
        return T_aug

    def _prepare_episode(self, traj: np.ndarray, pc_man_0: np.ndarray, pc_tgt_0: np.ndarray, augment: bool):
        pc_man_0 = resize_points(pc_man_0, self.num_pts)
        pc_tgt_0 = resize_points(pc_tgt_0, self.num_pts)
        centroid_0 = pc_man_0.mean(axis=0).astype(np.float32)

        pc_man_local = (pc_man_0 - centroid_0).astype(np.float32)
        pc_tgt_local = (pc_tgt_0 - centroid_0).astype(np.float32)

        local_traj = np.stack(
            [raw_pose7d_to_local_pose7d(pose, centroid_0) for pose in traj[:, :7]],
            axis=0
        ).astype(np.float32)
        local_traj = ensure_quat_continuity(local_traj)

        if self.resample_to_horizon:
            local_traj = resample_se3_trajectory(local_traj, self.horizon)
        elif local_traj.shape[0] != self.horizon:
            raise ValueError(
                f"Episode length {local_traj.shape[0]} != horizon {self.horizon}; "
                "enable resample_to_horizon=True."
            )

        if augment and self.mode == "train" and self.use_se3_aug:
            T_aug = self._get_se3_augmentation()
            T_aug_inv = np.linalg.inv(T_aug)

            pc_man_h = np.concatenate([pc_man_local, np.ones((self.num_pts, 1), dtype=np.float32)], axis=1)
            pc_tgt_h = np.concatenate([pc_tgt_local, np.ones((self.num_pts, 1), dtype=np.float32)], axis=1)
            pc_man_local = (T_aug @ pc_man_h.T).T[:, :3].astype(np.float32)
            pc_tgt_local = (T_aug @ pc_tgt_h.T).T[:, :3].astype(np.float32)

            aug_traj = []
            for pose in local_traj:
                T_i = pose7d_to_matrix_np(pose)
                T_i_aug = T_aug @ T_i @ T_aug_inv
                aug_traj.append(matrix_to_pose7d_np(T_i_aug))
            local_traj = ensure_quat_continuity(np.stack(aug_traj, axis=0))

        T_start = pose7d_to_matrix_np(local_traj[0])
        T_start_inv = np.linalg.inv(T_start)
        T_goal = pose7d_to_matrix_np(local_traj[-1])
        T_goal_delta = T_start_inv @ T_goal

        action = []
        for pose in local_traj:
            T_i = pose7d_to_matrix_np(pose)
            T_rel = T_start_inv @ T_i
            action.append(matrix_to_pose7d_np(T_rel))
        action = ensure_quat_continuity(np.stack(action, axis=0))

        identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        action[0] = identity
        goal_delta_pose7d = matrix_to_pose7d_np(T_goal_delta)
        action[-1] = goal_delta_pose7d

        goal_pose9d = matrix_to_pose9d_np(T_goal)
        goal_delta_pose9d = matrix_to_pose9d_np(T_goal_delta)

        return {
            "pc_manipulated": pc_man_local.astype(np.float32),
            "pc_target": pc_tgt_local.astype(np.float32),
            "local_traj": local_traj.astype(np.float32),
            "action": action.astype(np.float32),
            "agent_pos": local_traj[:1].astype(np.float32),
            "goal_pose9d": goal_pose9d.astype(np.float32),
            "goal_delta_pose7d": goal_delta_pose7d.astype(np.float32),
            "goal_delta_pose9d": goal_delta_pose9d.astype(np.float32),
            "centroid_0": centroid_0.astype(np.float32),
        }

    def get_normalizer(self, mode="limits", **kwargs):
        action_trans = []
        agent_pos_trans = []
        goal_pose9d = []
        goal_delta_pose9d = []

        for traj, pc_man_0, pc_tgt_0 in zip(self.trajectories, self.pc_man_0_list, self.pc_tgt_0_list):
            sample = self._prepare_episode(traj, pc_man_0, pc_tgt_0, augment=False)
            action_trans.append(sample["action"][:, :3])
            agent_pos_trans.append(sample["agent_pos"][:, :3])
            goal_pose9d.append(sample["goal_pose9d"])
            goal_delta_pose9d.append(sample["goal_delta_pose9d"])

        if len(action_trans) == 0:
            raise RuntimeError("FullTrajectoryGoalConditionedSE3Dataset is empty; check data_dirs.")

        data = {
            "action": np.concatenate(action_trans, axis=0).astype(np.float32),
            "agent_pos": np.concatenate(agent_pos_trans, axis=0).astype(np.float32),
            "goal_pose9d": np.stack(goal_pose9d, axis=0).astype(np.float32),
            "goal_delta_pose9d": np.stack(goal_delta_pose9d, axis=0).astype(np.float32),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        sample = self._prepare_episode(
            traj,
            self.pc_man_0_list[idx],
            self.pc_tgt_0_list[idx],
            augment=True,
        )

        obs = {
            "agent_pos": torch.from_numpy(sample["agent_pos"]).float(),
            "pc_manipulated": torch.from_numpy(sample["pc_manipulated"]).float().unsqueeze(0),
            "pc_target": torch.from_numpy(sample["pc_target"]).float().unsqueeze(0),
            "goal_pose9d": torch.from_numpy(sample["goal_pose9d"]).float().unsqueeze(0),
            "goal_delta_pose9d": torch.from_numpy(sample["goal_delta_pose9d"]).float().unsqueeze(0),
            "goal_delta_pose7d": torch.from_numpy(sample["goal_delta_pose7d"]).float().unsqueeze(0),
        }

        lang_emb = self.traj_lang_embs[idx] if self.use_lang_emb and len(self.traj_lang_embs) > idx else None
        if self.use_lang_emb and lang_emb is not None:
            lang_emb = np.asarray(lang_emb, dtype=np.float32)
            if lang_emb.ndim == 1:
                lang_emb = lang_emb[None, :]
            obs["lang_token_embs"] = torch.from_numpy(lang_emb).float()

        return {
            "obs": obs,
            "action": torch.from_numpy(sample["action"]).float(),
            "traj_idx": torch.tensor(idx, dtype=torch.long),
            "centroid_0": torch.from_numpy(sample["centroid_0"]).float(),
            "goal_delta_pose7d_debug": torch.from_numpy(sample["goal_delta_pose7d"]).float(),
            "goal_delta_pose9d_debug": torch.from_numpy(sample["goal_delta_pose9d"]).float(),
        }
