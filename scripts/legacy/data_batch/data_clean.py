import os
import glob
import numpy as np
import cv2

# ================= 配置区 =================
DST_ROOT = "/media/ljian/lj/data_3d/pouring"  # 你的数据根目录

# --- 阶段一：抗噪 SVD 参数 ---
VISIB_THRESHOLD = 0.5        # TAPIP3D 置信度阈值
OUTLIER_STD_MULTIPLIER = 1.5 # 离群点剔除阈值系数

# --- 阶段二：运动学降采样参数 ---
BASE_DOWNSAMPLE_RATE = 5     # 基础抽帧率
TAU_TRANS = 0.005            # 平移运动阈值 (0.005米 = 5mm)
TAU_ROT_DEG = 2.0            # 旋转运动阈值 (度)
TAU_ROT = np.radians(TAU_ROT_DEG) 

# --- 可视化参数 ---
AXIS_LENGTH = 0.005           # 坐标轴长度 5cm
# ==========================================

def compute_weighted_rigid_transform_se3(P, Q, weights):
    """
    带权重的 Kabsch 算法，输出 4x4 SE(3) 矩阵
    目标: Q_i = T @ P_i
    """
    w_sum = np.sum(weights)
    if w_sum < 1e-6:
        return np.eye(4)
    
    # 计算加权质心
    centroid_P = np.sum(P * weights[:, None], axis=0) / w_sum
    centroid_Q = np.sum(Q * weights[:, None], axis=0) / w_sum
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    H = P_centered.T @ np.diag(weights) @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_Q - R @ centroid_P
    
    T = np.eye(4)
    T[:3, :3] = R
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
    """
    将 3D 坐标系绘制到图像上。
    坐标系的原点被死死“锚定”在 centroid_ref 上，并受 T_matrix 驱动。
    """
    # 1. 在原点处定义一个标准的坐标系十字星
    axes_3d_local = np.array([
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale]
    ])
    
    # 2. 【核心锚定】将这个十字星平移到第一帧物体的质心上
    axes_3d_anchored = axes_3d_local + centroid_ref
    
    # 3. 转换为齐次坐标 (4, 4)
    axes_3d_anchored_h = np.concatenate([axes_3d_anchored, np.ones((4, 1))], axis=1)
    
    # 4. 施加相机系下的相对位姿变换 T_{0 -> t}
    axes_cam_t = (T_matrix @ axes_3d_anchored_h.T).T[:, :3]
    
    # 5. 投影到 2D 像素平面
    pts_2d = project_points(axes_cam_t, intrinsics)
    origin, pt_x, pt_y, pt_z = pts_2d[0], pts_2d[1], pts_2d[2], pts_2d[3]
    
    # 6. 画线 (OpenCV BGR: X红, Y绿, Z蓝)
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
        centroid_ref = np.mean(P_ref, axis=0) # 第一帧物理质心（锚点）
        
        out_dir = os.path.join(ep_path, "se3_trajectory")
        viz_dir = os.path.join(ep_path, "viz")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        # =========================================================
        # 第一阶段：强力抗噪 SVD 提取相对位姿 T_{0 -> t}
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
            
            # 初始求解
            T_init = compute_weighted_rigid_transform_se3(P_ref_v, P_t_v, W_t_v)
            
            # 计算重投影误差寻找内点 (Inliers)
            P_ref_v_h = np.concatenate([P_ref_v, np.ones((len(P_ref_v), 1))], axis=1)
            P_t_hat = (T_init @ P_ref_v_h.T).T[:, :3]
            
            errors = np.linalg.norm(P_t_v - P_t_hat, axis=1)
            thresh = np.mean(errors) + OUTLIER_STD_MULTIPLIER * np.std(errors)
            inlier_mask = errors < thresh
            
            # 二次精炼
            if np.sum(inlier_mask) >= 3:
                T_final = compute_weighted_rigid_transform_se3(
                    P_ref_v[inlier_mask], P_t_v[inlier_mask], np.ones(np.sum(inlier_mask))
                )
            else:
                T_final = T_init
                
            T_matrices[t] = T_final

        # =========================================================
        # 第二阶段：运动学锚点降采样
        # =========================================================
        base_idx = np.arange(0, T_frames, BASE_DOWNSAMPLE_RATE)
        kept_frames = [base_idx[0]]
        anchor = kept_frames[0]
        
        for i in range(1, len(base_idx)):
            curr = base_idx[i]
            T_anchor, T_curr = T_matrices[anchor], T_matrices[curr]
            
            # 提取相对平移和旋转矩阵
            delta_t = T_curr[:3, 3] - T_anchor[:3, 3]
            delta_R = T_anchor[:3, :3].T @ T_curr[:3, :3]
            
            dist = np.linalg.norm(delta_t)
            theta = np.arccos(np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0))
            
            if dist > TAU_TRANS or abs(theta) > TAU_ROT:
                kept_frames.append(curr)
                anchor = curr
                
        if kept_frames[-1] != base_idx[-1]:
            kept_frames.append(base_idx[-1])
            
        print(f"  [+] 降采样完成: {T_frames}帧 -> {len(kept_frames)}帧")

        # =========================================================
        # 第三阶段：单图叠影可视化与保存
        # =========================================================
        # 将结果保存为你仿真代码完美兼容的 (N, 4, 4) 格式
        final_trajectory = T_matrices[kept_frames]
        np.savez_compressed(
            os.path.join(out_dir, "se3_relative_trajectory.npz"),
            kept_indices=kept_frames,
            T_cam_0_to_t=final_trajectory 
        )
        
        # 可视化：提取第 0 帧作为静止画布
        viz_canvas = cv2.cvtColor(video[0], cv2.COLOR_RGB2BGR)
        cam_intrinsics = intrinsics[0]
        
        prev_pt2d = None
        for T_k in final_trajectory:
            # 将变换矩阵 T_k 施加在锚点坐标系上，并画在第一帧画布上
            curr_pt2d = draw_3d_axes_on_image(viz_canvas, T_k, cam_intrinsics, centroid_ref, scale=AXIS_LENGTH)
            
            if prev_pt2d is not None:
                cv2.line(viz_canvas, tuple(prev_pt2d), tuple(curr_pt2d), (0, 255, 255), 1)
            prev_pt2d = curr_pt2d

        viz_path = os.path.join(viz_dir, "se3_trajectory_overlay_on_first_frame.png")
        cv2.imwrite(viz_path, viz_canvas)
        print(f"  [✔] 轨迹矩阵与可视化图已保存。")

if __name__ == "__main__":
    main()