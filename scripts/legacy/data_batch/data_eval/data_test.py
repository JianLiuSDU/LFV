import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= 配置区 =================
# 你要测试的任务文件夹路径 (例如：drawer_open 或 pickNplace)
TASK_DIR = "/media/ljian/lj/data_3d/pickNplace"
# ==========================================

def evaluate_trajectory_overlap(task_dir):
    print(f"========== 正在评估任务: {os.path.basename(task_dir)} ==========")
    
    episode_dirs = sorted(glob.glob(os.path.join(task_dir, "episode_*")))
    if not episode_dirs:
        print("未找到 episode 文件夹，请检查路径。")
        return

    all_trajs_xyz = []

    # 1. 加载所有轨迹的平移部分 (X, Y, Z)
    for ep_path in episode_dirs:
        npz_path = os.path.join(ep_path, "se3_trajectory", "dp_action_trajectory.npz")
        if not os.path.exists(npz_path):
            continue
        
        data = np.load(npz_path)
        # 获取 8D action，提取前 3 维 [64, 3]
        actions_8d = data['actions_8d']
        xyz = actions_8d[:, :3]
        all_trajs_xyz.append(xyz)

    if not all_trajs_xyz:
        print("未找到有效的 dp_action_trajectory.npz 文件。")
        return

    # [N_episodes, 64, 3]
    all_trajs_xyz = np.array(all_trajs_xyz)
    N, T, D = all_trajs_xyz.shape
    print(f"成功加载 {N} 条轨迹，序列长度: {T}。")

    # 2. 计算量化重叠指标
    # (1) 计算每个时间步的“平均中心轨迹”
    mean_traj = np.mean(all_trajs_xyz, axis=0) # [64, 3]
    
    # (2) 计算每个时间步，所有轨迹距离平均轨迹的“欧氏距离”的标准散布 (单位：米)
    # 这代表了轨迹管子的“粗细”
    distances_to_mean = np.linalg.norm(all_trajs_xyz - mean_traj, axis=2) # [N, 64]
    mean_spread_per_step = np.mean(distances_to_mean, axis=0) # [64]
    max_spread_per_step = np.max(distances_to_mean, axis=0)   # [64]

    print("\n--- 轨迹散布量化报告 ---")
    print(f"起步 (t=0)  平均偏离中心距离: {mean_spread_per_step[0]:.6f} m (最大偏离: {max_spread_per_step[0]:.6f} m)")
    print(f"中段 (t={T//2}) 平均偏离中心距离: {mean_spread_per_step[T//2]:.6f} m (最大偏离: {max_spread_per_step[T//2]:.6f} m)")
    print(f"末端 (t={T-1}) 平均偏离中心距离: {mean_spread_per_step[-1]:.6f} m (最大偏离: {max_spread_per_step[-1]:.6f} m)")

    # 3. 3D 可视化绘制
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画出每一条真实轨迹 (半透明)
    for i in range(N):
        ax.plot(all_trajs_xyz[i, :, 0], 
                all_trajs_xyz[i, :, 1], 
                all_trajs_xyz[i, :, 2], 
                color='blue', alpha=0.15, linewidth=1)
                
    # 画出平均轨迹 (加粗红色)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2], 
            color='red', linewidth=3, label='Mean Trajectory')

    # 标记起点和终点
    ax.scatter([0], [0], [0], color='green', s=100, label='Start (0,0,0)', marker='o')
    ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], mean_traj[-1, 2], 
               color='black', s=100, label='Mean End', marker='x')

    ax.set_title(f"3D Trajectory Distribution for {os.path.basename(task_dir)}\n(N={N} episodes)", fontsize=14)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    
    # 保持三轴比例一致，避免视觉拉伸
    max_range = np.array([all_trajs_xyz[..., 0].max()-all_trajs_xyz[..., 0].min(),
                          all_trajs_xyz[..., 1].max()-all_trajs_xyz[..., 1].min(),
                          all_trajs_xyz[..., 2].max()-all_trajs_xyz[..., 2].min()]).max() / 2.0
    mid_x = (all_trajs_xyz[..., 0].max()+all_trajs_xyz[..., 0].min()) * 0.5
    mid_y = (all_trajs_xyz[..., 1].max()+all_trajs_xyz[..., 1].min()) * 0.5
    mid_z = (all_trajs_xyz[..., 2].max()+all_trajs_xyz[..., 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    # 保存图片到任务目录下
    save_path = os.path.join(task_dir, "trajectory_distribution_3d.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[+] 3D 分布图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 替换为你实际的任务文件夹路径
    tasks_to_eval = [
        #"/media/ljian/lj/data_3d/drawer_open",
        "/media/ljian/lj/data_3d/pickNplace" # 你可以把第二个任务也放进来一起跑
    ]
    
    for t_dir in tasks_to_eval:
        evaluate_trajectory_overlap(t_dir)