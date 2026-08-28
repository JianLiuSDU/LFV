# 单视角 RGB-D 的接触对抓取基础设施

## 目标与边界

在线端只假设有一张 RGB、一张深度图、相机内参和 Stage 1 输出的二维连续热力图。单视角反投影得到的点云记为 `P_vis`，它不包含遮挡面，因此不能把“补全”当成已观测事实。红色杯子仿真快照中的 `full_points_camera` 只用于离线评估，绝不参与候选生成。

SAM3D-Objects 作为可选的离线/A100 补全后端保留在 `lfv.geometry.sam3d_completion.SAM3DSubprocessBackend`。由于官方权重需要 Hugging Face gated 权限，当前闭环使用可复现的几何降级方案；获得权重后可通过现有 `NPZCompletionBackend`/SAM3D 后端替换，不需要改变 Stage 1 或执行接口。

## 三种后端路径

1. **可见点云基线**：`VisibleOnlyCompletionBackend`，只把掩码内有效深度反投影。它适合检查 RGB-D、坐标系和热力图对齐，不声称拥有另一侧几何。
2. **接触对/厚度假设（当前默认）**：`build_contact_pair_hypotheses` 在高热区域求加权中心，使用可见点云 PCA 与相机水平轴生成少量夹持轴，并枚举夹爪宽度。第一接触点必须由可见点支持，第二接触点是沿夹持轴的虚拟对侧点。TCP 的 `z` 轴固定为首选 top-down 方向，`y` 轴为夹持方向，`x` 轴由叉积构造，保证右手系。该方法不需要对称物体的 OBB/PCA 坐标作为论文假设；PCA 仅用于少量候选提议，并可替换为学习式/GraspNet 后端。
3. **完整或模板点云 GraspNet**：`ExternalGraspNetBackend` 可调用外部 GraspNet 命令；若获得离线 SAM3D/模板点云，先用 `NPZCompletionBackend` 或 ICP 对齐，再交给 GraspNet。旧的 full-cloud GraspNet 文件不能被误当作 partial-cloud 结果，当前脚本只将其作为未来接口，不默认读取。

## 计算流程

```text
RGB-D + cup mask + Stage1 heatmap
        │
        ├─ 有效像素反投影 → P_vis (camera frame)
        ├─ heatmap[uv_vis] → h_vis
        └─ 接触对候选：
             heat-weighted center → in-plane axes → width sweep
             observed contact p, virtual opposite q
             right-handed top-down TCP
        │
        ├─ online: 输出候选和 selected_grasp_partial.npz
        └─ simulator-only: 用 full_points_camera 做最近邻距离/双接触率评估
```

候选得分为 `0.60 * heat_peak + 0.25 * endpoint_support + 0.15 * topdown_alignment`。这是工程上的可解释排序，不是训练损失。离线评估报告第一、第二接触点到完整仿真点云的最近距离、是否双侧支持、夹持宽度和 top-down 对齐度。

## 可复现实验

```bash
cd /home/users1/ljian/LFV_stage2_motion_field
PYTHONPATH=. /home/users1/ljian/anaconda3/envs/da3/bin/python \
  scripts/deployment/run_red_cup_partial_grasp.py
```

默认输入是：

* 快照：`/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/red_mug_a3b_seed_0/snapshot/pouring_snapshot.npz`；
* Stage 1 迁移：同目录 `transfer/transfer_result.npz` 的 `target_heatmap`；
* 输出：同目录 `partial_grasp/`。

主要输出：

* `selected_grasp_partial.npz`：相机坐标系 TCP、第一/第二接触点、夹持轴、宽度；
* `contact_pair_hypotheses.npz`：全部候选；
* `partial_grasp_overlay.png`：热力图、红色可见接触、蓝色虚拟对侧接触和 TCP 三轴；
* `partial_grasp_report.json`：输入来源、是否使用完整点云评估、候选质量和限制。

红杯快照的当前离线评估中，选中候选的 top-down 对齐度为 1.0，第一/第二接触到仿真完整点云的最近距离约为毫米级；这个结果只证明“接触对假设在该仿真实例上可行”，不等价于真实机器人碰撞安全。真实执行前仍需 GraspNet/碰撞检测、夹爪宽度限制、IK 和工作空间检查。

## 与 Stage 1/Stage 2 的接口

Stage 1 仍只负责 `target_heatmap`（可见区域热力）；本模块把它采样到 `visible_pixels_uv`，不改变热力迁移算法。Stage 2 仍接收原有相机/对象轨迹接口；抓取选择输出的 `tcp_camera` 可以直接作为对象到 TCP attachment 的起点。后续替换为 SAM3D、PoinTr/P2C 或真实 partial-cloud GraspNet 时，只需实现同样的 `complete(...)` 或候选选择接口。

