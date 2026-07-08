import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- 处理 Zarr 编解码器 ---
import numcodecs
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
except ImportError:
    pass

import zarr

# ================= 配置区 =================
DST_ROOT = "/media/ljian/lj/data_3d/pouring"

NUM_SAMPLES = 256
# ==========================================

def uniform_grid_sampling(mask, bbox, num_samples):
    """
    在 BBox 范围内生成网格，并保留落在 Mask 内部的点。
    如果点数过多则随机下采样，如果点数过少则允许重复采样补齐。
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    # 动态计算网格大小，确保初始网格点数量略大于目标采样数
    area = (x_max - x_min) * (y_max - y_min)
    grid_size = int(np.sqrt(area / (num_samples * 2)))
    grid_size = max(1, grid_size) # 防止 grid_size 为 0

    xs = np.arange(x_min, x_max, grid_size)
    ys = np.arange(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # 过滤出真正在 Mask 内部的点
    valid_points = []
    for p in points:
        x, y = int(p[0]), int(p[1])
        # 边界安全检查
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x]:
                valid_points.append([x, y])
    
    valid_points = np.array(valid_points)
    
    if len(valid_points) == 0:
        raise ValueError("Mask 内部没有采样到任何有效网格点！请检查 Mask 是否为空。")
    
    # 调整到目标数量 (NUM_SAMPLES)
    if len(valid_points) > num_samples:
        # 点多了，无放回随机抽样
        indices = np.random.choice(len(valid_points), num_samples, replace=False)
        valid_points = valid_points[indices]
    elif len(valid_points) < num_samples:
        # 点少了，有放回随机抽样来补齐
        pad_size = num_samples - len(valid_points)
        pad_indices = np.random.choice(len(valid_points), pad_size, replace=True)
        valid_points = np.vstack([valid_points, valid_points[pad_indices]])
        
    return valid_points

def main():
    episode_dirs = sorted(glob.glob(os.path.join(DST_ROOT, "episode_*")))
    total_eps = len(episode_dirs)
    print(f"[*] 发现 {total_eps} 个 episode，开始批量 2D 均匀采样...")

    failed_episodes = []

    for idx, ep_path in enumerate(episode_dirs):
        ep_name = os.path.basename(ep_path)
        print(f"--- 进度 [{idx+1}/{total_eps}]: {ep_name} ---")

        # 检查前置文件
        bbox_path = os.path.join(ep_path, "target_bbox", "target_bbox.npy")
        mask_path = os.path.join(ep_path, "target_sam_mask", "target_mask.npy")
        
        if not (os.path.exists(bbox_path) and os.path.exists(mask_path)):
            print(f"  [!] 跳过: 未找到 BBox 或 Mask 文件。")
            failed_episodes.append(ep_name)
            continue

        try:
            # 1. 加载数据
            bbox = np.load(bbox_path)
            mask = np.load(mask_path)
            
            rgb_zarr_path = os.path.join(ep_path, "rgb")
            episode_rgb = zarr.open(rgb_zarr_path, mode="r")
            initial_frame = episode_rgb[0]

            # 2. 建立专属输出目录
            sample_dir = os.path.join(ep_path, "target_sample_points")
            viz_dir = os.path.join(ep_path, "viz")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 3. 执行采样
            points_2d = uniform_grid_sampling(mask, bbox, NUM_SAMPLES)
            
            # 4. 保存采样点数据
            # 使用字典包裹保存，方便未来扩展（比如后续可能加入 normals 或 weights）
            save_dict = {'query_points_2d': points_2d}
            npy_path = os.path.join(sample_dir, "target_sampled_2d_uniform.npy")
            np.save(npy_path, save_dict)

            # 5. 可视化保存
            viz_save_path = os.path.join(viz_dir, "target_sampling_2d_uniform.png")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(initial_frame)
            # 画出 mask 的红色半透明轮廓辅助观察
            ax.contour(mask, colors='red', linewidths=1.0, alpha=0.6)
            # 散点图画出采样的 2D 点，使用亮绿色 (lime) 对比度更高
            ax.scatter(points_2d[:, 0], points_2d[:, 1], c='lime', s=15, marker='o', edgecolors='black', linewidths=0.5)
            
            ax.set_title(f"{ep_name} | Uniform Sampling (N={NUM_SAMPLES})")
            ax.axis('off')
            fig.savefig(viz_save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)

            print(f"  [✔] 成功采样 {NUM_SAMPLES} 个点！")

        except Exception as e:
            print(f"  [✘] 异常: {str(e)}")
            failed_episodes.append(ep_name)

    # 总结
    print("\n================ 批量 2D 采样完成 ================")
    print(f"成功: {total_eps - len(failed_episodes)} / {total_eps}")
    if failed_episodes:
        print(f"失败的 Episodes: {failed_episodes}")
        
    print(f"\n可以通过以下命令批量查看采样结果：")
    print(f"ls -v {DST_ROOT}/episode_*/viz/target_sampling_2d_uniform.png | xargs eog")

if __name__ == "__main__":
    main()