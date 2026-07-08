import os
import glob
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

# ================= 配置区 =================
DST_ROOT = "/media/ljian/lj/data_3d/pouring"

# --- 阶段一：抗噪 SVD 参数 ---
VISIB_THRESHOLD = 0.5        
OUTLIER_STD_MULTIPLIER = 1.5 

# --- 阶段二：扩散策略 (DP) 重采样参数 ---
DP_HORIZON = 64              # 扩散策略要求的固定序列长度 N
LAMBDA_ROT = 0.1             # 计算路径长度时，旋转(弧度)的权重系数 (1rad 约等于 10cm 平移)

# --- 可视化参数 ---
AXIS_LENGTH = 0.005          # 坐标轴长度 (根据你的单位可能是 5mm)
# ==========================================

def compute_weighted_rigid_transform_se3(P, Q, weights):
    """带权重的 Kabsch 算法，输出 4x4 SE(3) 矩阵"""
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
        
    t = centroid_Q - R_mat @ centroid_P
    
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def project_points(pts_3d, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = np.maximum(pts_3d[:, 2], 1e-5)
    x2d = (pts_3d[:, 0] * fx / z) + cx
    y2d = (pts_3d[:, 1] * fy / z) + cy
    return np.stack([x2d, y2d], axis=-1).astype(int)

def draw_3d_axes_on_image(img, T_matrix, intrinsics, centroid_ref, scale=0.05):
    """将 3D 坐标系绘制到图像上，锚定在 centroid_ref 上"""
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

def main():
    episode_dirs = sorted(glob.glob(os.path.join(DST_ROOT, "episode_*")))
    
    for ep_path in episode_dirs:
        ep_name = os.path.basename(ep_path)
        npz_path = os.path.join(ep_path, "point_tracking", "tapip3d_result.npz")
        if not os.path.exists(npz_path):
            continue
            
        print(f"\n--- 正在处理 {ep_name} ---")
        
        data = np.load(npz_path)
        coords = data['coords']
        visibs = data['visibs']
        video = data['video']
        intrinsics = data['intrinsics']
        
        T_frames = coords.shape[0]
        P_ref = coords[0]
        centroid_ref = np.mean(P_ref, axis=0) 
        
        out_dir = os.path.join(ep_path, "se3_trajectory")
        viz_dir = os.path.join(ep_path, "viz")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        # =========================================================
        # 第一阶段：强力抗噪 SVD 提取稠密 SE(3)
        # =========================================================
        T_matrices = np.zeros((T_frames, 4, 4))
        T_matrices[0] = np.eye(4)
        
        for t in range(1, T_frames):
            P_t, W_t = coords[t], visibs[t]
            valid_mask = W_t > VISIB_THRESHOLD
            
            if np.sum(valid_mask) < 3:
                T_matrices[t] = T_matrices[t-1]
                continue
                
            P_ref_v, P_t_v, W_t_v = P_ref[valid_mask], P_t[valid_mask], W_t[valid_mask]
            
            T_init = compute_weighted_rigid_transform_se3(P_ref_v, P_t_v, W_t_v)
            
            P_ref_v_h = np.concatenate([P_ref_v, np.ones((len(P_ref_v), 1))], axis=1)
            P_t_hat = (T_init @ P_ref_v_h.T).T[:, :3]
            
            errors = np.linalg.norm(P_t_v - P_t_hat, axis=1)
            thresh = np.mean(errors) + OUTLIER_STD_MULTIPLIER * np.std(errors)
            inlier_mask = errors < thresh
            
            if np.sum(inlier_mask) >= 3:
                T_final = compute_weighted_rigid_transform_se3(
                    P_ref_v[inlier_mask], P_t_v[inlier_mask], np.ones(np.sum(inlier_mask))
                )
            else:
                T_final = T_init
                
            T_matrices[t] = T_final

        # =========================================================
        # 第二阶段：空间物理路程重采样 (Diffusion Policy 对齐)
        # =========================================================
        # 1. 拆解为平移和四元数 (x, y, z, qx, qy, qz, qw)
        pos = T_matrices[:, :3, 3] # [T, 3]
        quats = R_scipy.from_matrix(T_matrices[:, :3, :3]).as_quat() # [T, 4] scipy 默认顺序是 x,y,z,w
        
        # 2. 强制四元数连续性对齐 (极其重要，防止符号翻转导致 DP 崩溃)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
                
        # 3. 计算物理累积路径长度 S
        delta_pos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        
        # 计算相邻两帧的旋转角度变化
        dots = np.sum(quats[1:] * quats[:-1], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        delta_theta = 2 * np.arccos(np.abs(dots)) # 使用绝对值保证物理旋转角为正
        
        # 综合距离 = 平移距离 + 权重 * 旋转角度
        delta_s = delta_pos + LAMBDA_ROT * delta_theta
        S = np.concatenate([[0], np.cumsum(delta_s)])
        
        # 注入微小噪声保证 S 严格单调递增 (解决人类完全不动时导致插值库报错)
        S = S + np.linspace(0, 1e-7, len(S))
        
        # 4. 生成定长的目标路点
        S_target = np.linspace(0, S[-1], DP_HORIZON)
        
        # 5. 三次样条插值平移
        pos_interp = interp1d(S, pos, axis=0, kind='cubic')(S_target)
        
        # 6. Slerp 球面线性插值旋转
        slerp = Slerp(S, R_scipy.from_quat(quats))
        quats_interp = slerp(S_target).as_quat()
        
        # 7. 夹爪状态：前 63 帧闭合 (0)，最后一帧张开 (1)
        gripper_state = np.zeros((DP_HORIZON, 1), dtype=np.float32)
        gripper_state[-1, 0] = 1.0
        
        # 组合为 DP 训练所需的 8D 动作张量 [N, 8]
        action_8d_trajectory = np.concatenate([pos_interp, quats_interp, gripper_state], axis=1)

        # 逆向重构回 [N, 4, 4] 仅供可视化使用
        T_matrices_resampled = np.zeros((DP_HORIZON, 4, 4))
        T_matrices_resampled[:, 3, 3] = 1.0
        T_matrices_resampled[:, :3, 3] = pos_interp
        T_matrices_resampled[:, :3, :3] = R_scipy.from_quat(quats_interp).as_matrix()

        print(f"  [+] DP 空间重采样完成: {T_frames}帧 -> {DP_HORIZON}步固定动作序列 (8D 包含夹爪)")

        # =========================================================
        # 第三阶段：保存 8D 动作与可视化
        # =========================================================
        np.savez_compressed(
            os.path.join(out_dir, "dp_action_trajectory.npz"),
            actions_8d=action_8d_trajectory,       # [64, 8] 送入 Diffusion Policy 训练的数据 (唯一改变内部键名为 actions_8d 以匹配维度)
            T_matrices_4x4=T_matrices_resampled    # [64, 4, 4] 方便你兼容旧代码流
        )
        
        viz_canvas = cv2.cvtColor(video[0], cv2.COLOR_RGB2BGR)
        cam_intrinsics = intrinsics[0]
        
        prev_pt2d = None
        # 此时画出的是重采样后完美的 64 个间距均匀的坐标系
        for T_k in T_matrices_resampled:
            curr_pt2d = draw_3d_axes_on_image(viz_canvas, T_k, cam_intrinsics, centroid_ref, scale=AXIS_LENGTH)
            
            if prev_pt2d is not None:
                cv2.line(viz_canvas, tuple(prev_pt2d), tuple(curr_pt2d), (0, 255, 255), 1)
            prev_pt2d = curr_pt2d

        viz_path = os.path.join(viz_dir, "dp_trajectory_overlay.png")
        cv2.imwrite(viz_path, viz_canvas)
        print(f"  [✔] 8D 动作序列与可视化图已保存并覆盖。")

if __name__ == "__main__":
    main()