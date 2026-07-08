import os
import glob
import torch
import pickle
import numpy as np
import cv2

# --- 解决 Zarr 编解码 ---
import numcodecs
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
except ImportError:
    pass
import zarr

# --- 导入 TAPIP3D 依赖 ---
from utils.inference_utils import load_model, inference

# ================= 配置区 =================
DST_ROOT = "/media/ljian/lj/data_3d/pouring"
INTRINSICS_PATH = "/home/users1/ljian/im2Flow2Act/data_local/simulation/instrinsic_5-1.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAPIP3D_CHECKPOINT = "checkpoints/tapip3d_final.pth"

SUPPORT_GRID_SIZE = 0
NUM_ITERS = 6
VISIB_THRESHOLD = 0.8
# ==========================================

def load_intrinsics():
    """加载并解析相机内参"""
    with open(INTRINSICS_PATH, 'rb') as f:
        intrinsics_raw = pickle.load(f) 
        
    if isinstance(intrinsics_raw, dict):
        fx = intrinsics_raw.get('fx', intrinsics_raw.get('f'))
        fy = intrinsics_raw.get('fy', intrinsics_raw.get('f'))
        cx = intrinsics_raw.get('cx')
        cy = intrinsics_raw.get('cy')
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        intrinsics = intrinsics_raw.copy()
    return intrinsics

def project_to_2d(pts_3d, intrinsics):
    """将 3D 点云投影回 2D 像素坐标 (用于视频可视化)"""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    z = np.maximum(pts_3d[:, 2], 1e-5) 
    x2d = (pts_3d[:, 0] * fx / z) + cx
    y2d = (pts_3d[:, 1] * fy / z) + cy
    return np.stack([x2d, y2d], axis=-1).astype(int)

def main():
    print(f"[*] 正在加载 TAPIP3D 模型 (仅加载一次)...")
    model = load_model(TAPIP3D_CHECKPOINT)
    model.to(DEVICE)
    print("[✔] 模型加载完毕！")

    intrinsics_base = load_intrinsics()
    fx, fy = intrinsics_base[0, 0], intrinsics_base[1, 1]
    cx, cy = intrinsics_base[0, 2], intrinsics_base[1, 2]

    episode_dirs = sorted(glob.glob(os.path.join(DST_ROOT, "episode_*")))
    total_eps = len(episode_dirs)
    failed_episodes = []

    for idx, ep_path in enumerate(episode_dirs):
        ep_name = os.path.basename(ep_path)
        print(f"\n--- 进度 [{idx+1}/{total_eps}]: 正在全量跟踪 {ep_name} ---")

        rgb_path = os.path.join(ep_path, "rgb")
        depth_path = os.path.join(ep_path, "depth")
        sample_path = os.path.join(ep_path, "sample_points", "sampled_2d_uniform.npy")
        
        if not (os.path.exists(sample_path) and os.path.exists(rgb_path) and os.path.exists(depth_path)):
            print(f"  [!] 数据不全，跳过。")
            failed_episodes.append(ep_name)
            continue

        try:
            # 1. 加载基础数据
            video_data = zarr.open(rgb_path, mode='r')[:]
            depth_data = zarr.open(depth_path, mode='r')[:]
            pts_2d = np.load(sample_path, allow_pickle=True).item()['query_points_2d']
            
            T, H, W, C = video_data.shape
            depth_0 = depth_data[0]
            
            # 2. 2D 采样点反投影到 3D
            pts_3d_cam = []
            for p in pts_2d:
                x, y = int(p[0]), int(p[1])
                d = depth_0[y, x]
                if d <= 0 or np.isnan(d): continue
                z_c = float(d)
                x_c = (x - cx) * z_c / fx
                y_c = (y - cy) * z_c / fy
                pts_3d_cam.append([x_c, y_c, z_c])
            pts_3d_cam = np.array(pts_3d_cam)

            if len(pts_3d_cam) == 0:
                raise ValueError("深度异常，未获得有效的 3D 初始点。")

            # 构造模型输入所需的 4D 查询点
            query_points = np.zeros((len(pts_3d_cam), 4), dtype=np.float32)
            query_points[:, 0] = 0          
            query_points[:, 1:] = pts_3d_cam

            # 3. 构造完美的相机内外参序列 (供推理和全量保存使用)
            intrinsics_seq = np.tile(intrinsics_base, (T, 1, 1))
            extrinsics_seq = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))

            # 4. 转换为 Tensor 并执行推理
            video_t = (torch.from_numpy(video_data).permute(0, 3, 1, 2).float() / 255.0).to(DEVICE)
            depths_t = torch.from_numpy(depth_data).float().to(DEVICE)
            intrinsics_t = torch.from_numpy(intrinsics_seq).float().to(DEVICE)
            extrinsics_t = torch.from_numpy(extrinsics_seq).float().to(DEVICE)
            query_point_t = torch.from_numpy(query_points).float().to(DEVICE)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                coords, visibs = inference(
                    model=model,
                    video=video_t,
                    depths=depths_t,
                    intrinsics=intrinsics_t,
                    extrinsics=extrinsics_t,
                    query_point=query_point_t,
                    num_iters=NUM_ITERS,
                    grid_size=SUPPORT_GRID_SIZE,
                )
            
            coords = coords.cpu().numpy()
            visibs = visibs.cpu().numpy()
            
            # 5. 【核心修改】全量保存所有上下文信息，保证下游 100% 兼容
            out_dir = os.path.join(ep_path, "point_tracking")
            viz_dir = os.path.join(ep_path, "viz")
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)
            
            npz_save_path = os.path.join(out_dir, "tapip3d_result.npz")
            np.savez_compressed(
                npz_save_path,
                video=video_data,         # 原始 RGB 视频
                depths=depth_data,        # 原始深度
                intrinsics=intrinsics_seq,# 完整的内参序列 [T, 3, 3]
                extrinsics=extrinsics_seq,# 完整的外参序列 [T, 4, 4]
                coords=coords,            # 追踪轨迹 [T, N, 3]
                visibs=visibs,            # 可见性置信度 [T, N]
                query_points=query_points # 初始查询点
            )

            # 6. 生成直观的 MP4 可视化视频 (保持不变)
            mp4_path = os.path.join(viz_dir, "tracking_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10 
            out = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))

            num_points = coords.shape[1]
            colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)

            for t in range(T):
                frame_bgr = cv2.cvtColor(video_data[t], cv2.COLOR_RGB2BGR)
                pts_2d_t = project_to_2d(coords[t], intrinsics_base)
                
                for p_idx in range(num_points):
                    if visibs[t, p_idx] > VISIB_THRESHOLD:
                        x2d, y2d = pts_2d_t[p_idx]
                        if 0 <= x2d < W and 0 <= y2d < H:
                            color = tuple(int(c) for c in colors[p_idx])
                            cv2.circle(frame_bgr, (x2d, y2d), radius=4, color=color, thickness=-1)
                            cv2.circle(frame_bgr, (x2d, y2d), radius=4, color=(0,0,0), thickness=1)

                out.write(frame_bgr)
            
            out.release()
            print(f"  [✔] {ep_name} 跟踪并全量打包完成！")

        except Exception as e:
            print(f"  [✘] 异常: {str(e)}")
            failed_episodes.append(ep_name)

    print("\n================ 批量 TAPIP3D 全量重采样完成 ================")
    print(f"成功: {total_eps - len(failed_episodes)} / {total_eps}")
    
if __name__ == "__main__":
    main()