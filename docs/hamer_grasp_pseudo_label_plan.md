# HaMeR Thumb-Index Grasp Pseudo Label Plan

## 结论

当前更推荐的版本是：不使用掌心、手腕、MCP 点去完整构造 6D 抓取姿态，而是用 HaMeR 的拇指和食指关键点提取“人实际捏/夹的位置”，再结合 RGB-D 物体点云、物体坐标系和机器人夹爪约束生成平行夹爪伪标签。

这样更符合本项目目标：从人类视频中提取任务相关抓取区域和机器人可用的抓取先验，而不是做人手姿态到机器人手的完整 retargeting。

## 为什么不默认用掌心法向

掌心法向理论上能提供 approach direction，但在当前数据中不是最稳的选择：

- HaMeR 是单目 hand mesh recovery，掌面 3D 法向容易受深度尺度、遮挡和手型估计误差影响。
- 抓杯把、杯身或抽屉把手时，机器人夹爪真正需要的是两个接触侧面和闭合方向，不一定需要复制人手掌面朝向。
- 掌面法向存在符号翻转问题，同一只手在相邻帧可能出现 approach 方向突变。
- 机器人通常有固定的工作空间和 TCP 约束，接近方向更应来自机器人可达性、任务先验、物体表面法向或 top-down 约束。

因此第一版不把掌心法向作为主逻辑。它只作为后续可选的姿态一致性检查，不参与默认位姿构造。

## ViGen 论文中的构造和迁移方式

本地论文文件：

```text
/home/users1/ljian/Downloads/TRO-26-0860_01_MS.pdf
```

论文标题是 `ViGen: Bridging Videos to Robot Manipulation via 3DGS Data Generation and Privileged Learning`。

其抓取构造核心在 Sec. IV-E `Multi-Constrained Grasp Optimization`：

- 先从人类视频中抽取 grasping frame。
- 用 HaMeR 估计 21 个手部 2D keypoints。
- 用深度图和相机内参把场景重建到一个虚拟 world 坐标系。
- 通过平面拟合得到支撑平面，把该平面作为 xy-plane。
- 假设机器人采用 top-down grasp，因此不估计任意 6D 姿态，而是把问题降成：
  - 3D grasp center；
  - 绕 z 轴的一维旋转 yaw。
- 将拇指和食指 keypoints 投影到 3D。
- 二者中点作为初始抓取中心。
- 二者连线在 xy 平面的投影决定抓取 yaw。
- 再结合物体空间位姿，把初始抓取位姿转换成物体坐标系下的 `T_grasp`。

论文没有直接信任初始结果，而是继续做几何优化：

- 在物体模型上根据当前 grasp center 和 yaw 计算左右夹爪接触点 `p_left`, `p_right`。
- 用左右接触点关于 grasp center 的对称性作为损失。
- 用接触点法向和夹爪闭合方向的一致性作为稳定性损失。
- 用 differential evolution 优化 grasp center 和 yaw。
- 最终输出 `T_grasp` 和夹爪宽度 `D_grasp = |p_left - p_right|`。

其迁移方式：

- 抓取位姿保存在源物体坐标系下。
- 新物体通过功能部件分割和对象对齐建立与源物体的对应关系。
- 对 grasp transfer，论文用源/目标物体对应部件之间的 affine transformation 映射 `T_grasp`。
- 对 interaction transfer，论文计算交互阶段 contact region center 作为 functional interaction point，再根据它在 3D bounding box 中的相对位置迁移到新物体，并平移整个 interaction trajectory。

对 LFV 的启发：

- 我们不需要一开始估完整 arbitrary 6D 人手抓取姿态。
- 更应该先把 thumb-index 变成物体表面上的两个接触点。
- 抓取位姿应保存在初始被操作物体坐标系中，而不是只保存在相机坐标系中。
- 后续迁移时，应围绕功能接触区域或接触点做对象内坐标迁移。

## 本项目推荐计算方法

### 输入

对每条 episode 复用已有结果：

- `contact_timing/contact_timing.json`
  - `anchor_frame`
  - `contact_start`
  - `contact_end`
  - `contact_frames`
- `rgb`, `depth`
- `meta.json` 中的 D455 内参和 `depth_scale`
- `hand_mask/frame_*.npy`
- `sam_mask/affordance_mask.npy` 或已有 object masks
- `contact_heatmap/contact_heatmap.npz`
  - `points_camera`
  - `points_object_m`
  - `pixels_uv`
  - `normals_camera`
  - `contact_heat`
  - `object_center_camera`
- 可选：`se3_trajectory/se3_relative_trajectory.npz`

### 1. 稳定抓取窗口

从已确定的接触时刻之后选 4 到 6 帧。第一版建议：

```text
window = contact_start 后最近的 4 帧
```

过滤条件：

- 手物仍处于接触状态；
- thumb-index 像素距离变化不过大；
- thumb/index 投影到物体表面的距离不过大；
- 候选 grasp center 和 yaw 在窗口内稳定。

### 2. HaMeR 推理

导出窗口 RGB 帧：

```text
<episode>/hamer_input/frame_000045.png
```

通过独立 HaMeR 环境运行：

```bash
cd /home/users1/ljian/LFV
bash scripts/run_hamer_demo_env.sh \
  --img_folder <episode>/hamer_input \
  --out_folder <episode>/hamer_output \
  --batch_size 8 \
  --save_skeleton2d \
  --full_frame
```

LFV 中的 HaMeR 入口：

```text
/home/users1/ljian/LFV/third_party/hamer -> /home/users1/ljian/hamer
/home/users1/ljian/anaconda3/envs/hamer/bin/python
```

读取：

```text
hamer_output/hamer_keypoints/*_pred_keypoints_2d.npy
hamer_output/hamer_keypoints/*_pred_keypoints_3d.npy
```

关键点索引：

```text
0: wrist
4: thumb_tip
8: index_tip
```

第一版只强依赖 `thumb_tip` 和 `index_tip`。

### 3. RGB-D 对齐

HaMeR 的 3D 输出不直接作为米制坐标。默认做法：

1. 取 HaMeR 的 2D thumb/index 像素。
2. 在 D455 深度图对应位置的局部窗口取有效深度中位数。
3. 用相机内参反投影为米制相机坐标。
4. 如果指尖深度无效，则用附近 hand mask 内有效深度或直接跳过该帧。

反投影：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth(u, v)
```

### 4. 投影到物体表面

对每帧的 thumb/index 米制点，查最近物体表面点：

```text
q_thumb = nearest_object_surface(thumb_tip_3d)
q_index = nearest_object_surface(index_tip_3d)
```

若已有物体 6D 轨迹：

- 把 anchor 物体点云变换到当前帧后查询；
- 或把当前帧 thumb/index 反变换回 anchor 物体坐标系后查询。

若暂时没有可靠 6D：

- 第一版只用 anchor 附近的接触窗口，帧数保持少；
- 对每帧直接用当前可见物体 mask + depth 生成临时物体点云进行查询；
- 最终把接触点转换回 anchor 物体坐标系。

### 5. 构造平行夹爪候选

默认使用 top-down / task-constrained grasp，不用掌心。

在物体坐标系中：

```text
center = 0.5 * (q_thumb + q_index)
closing = normalize(q_index - q_thumb)
width = norm(q_index - q_thumb) + width_margin
```

接近方向有三种可配置模式：

1. `top_down`
   - 适合桌面场景；
   - approach 固定为桌面法向反方向或相机/world z 方向。
2. `surface_normal`
   - 用两个接触点附近物体法向平均，取反作为接近方向；
   - 适合杯把、把手等非水平表面。
3. `camera_to_object`
   - 从相机朝物体中心方向接近；
   - 适合单视角可视化和候选初始化。

第一版建议默认：

```text
approach_mode = surface_normal
fallback = camera_to_object
```

正交化：

```text
closing = normalize(closing)
approach = normalize(approach - dot(approach, closing) * closing)
binormal = normalize(cross(approach, closing))
R = [approach, closing, binormal]
tcp = center - approach * tcp_to_contact_offset
```

这里沿用已有 GraspNet 可视化脚本中的约定：

```text
X = approach
Y = closing
Z = binormal
```

### 6. 窗口内选择代表姿态

每帧生成一个候选，然后过滤：

- `finger_surface_distance < 0.03m`
- `width_m` 在真实夹爪范围内，例如 `[0.015, 0.085]`
- `center` 位于 contact heat 高值区域附近；
- 相邻帧 yaw 跳变小；
- 相邻帧 center 跳变小；
- 简单夹爪线框不明显穿过物体主体。

代表姿态：

- 计算每个候选与窗口内其他候选的距离；
- 距离包括 center、yaw、width；
- 选择总距离最小的 median candidate；
- confidence 由有效帧比例、指尖-表面距离、候选一致性和 heat 匹配分数综合得到。

## 输出格式

保存到：

```text
<episode>/hamer_grasp_pseudo_label/
```

文件：

```text
grasp_pseudo_label.npz
grasp_pseudo_label_meta.json
viz/grasp_overlay_2d.png
viz/grasp_open3d_preview.png
viz/window_candidates.png
```

`npz` 字段：

```text
T_grasp_cam: (4, 4)
T_grasp_object: (4, 4)
rotation_6d: (6,)
translation_object: (3,)
width_m: scalar
q_thumb_object: (3,)
q_index_object: (3,)
candidate_T_object: (N, 4, 4)
candidate_width_m: (N,)
candidate_frames: (N,)
valid: bool
confidence: scalar
```

## 修改流程

### 阶段一：episode_0 最小闭环

新增：

- `lfv/pipeline/hamer_hand_pose.py`
  - 导出接触窗口帧；
  - 调用 `scripts/run_hamer_demo_env.sh`；
  - 读取和标准化 HaMeR 关键点输出。
- `lfv/pipeline/thumb_index_grasp_label.py`
  - 读取 RGB-D、关键点、物体点云、contact heat；
  - thumb/index 深度对齐；
  - 最近物体表面点查询；
  - 生成 top-down 或 surface-normal grasp；
  - 保存伪标签。
- `tools/visualize_thumb_index_grasp_label.py`
  - Open3D 显示物体点云、contact heat、thumb/index 接触点和夹爪。

修改：

- `configs/pipeline/hand_pouring.yaml`
  - 增加 `hamer` 和 `thumb_index_grasp` 配置。
- `scripts/run_pipeline.py`
  - 增加可选 stage：`hamer`, `thumb_index_grasp`。

### 阶段二：质量增强

- 自动稳定窗口选择；
- 多手/多候选选择；
- 用 SE(3) 轨迹统一到 anchor 物体坐标系；
- 加入物体法向的接近方向和简单碰撞过滤；
- 输出每帧候选姿态一致性报告。

### 阶段三：迁移与 GraspNet 对接

- 将 `T_grasp_object` 作为任务相关抓取先验；
- 用 contact heat 限制 GraspNet 候选区域；
- 用 thumb-index 伪标签重排序 GraspNet 候选：
  - center 靠近伪标签；
  - closing axis 对齐；
  - width 接近；
  - 接触点落在高热区域；
  - 无明显碰撞。

## 最小验证标准

只验证 `episode_0`：

1. HaMeR 2D 骨架中的 thumb/index 指尖正确落在人手上。
2. thumb/index 深度反投影点与场景尺度一致。
3. `q_thumb`, `q_index` 落在杯把或杯身真实接触区域附近。
4. 夹爪宽度合理。
5. Open3D 中可以拖动查看：
   - 杯子点云；
   - contact heat；
   - thumb/index 两个接触点；
   - 小夹爪线框；
   - TCP 坐标系。
6. JSON 中明确标记 `good/review/reject`，不自动删除失败样本。

## 当前已经完成的接入

- 已建立：

```text
/home/users1/ljian/LFV/third_party/hamer -> /home/users1/ljian/hamer
```

- 已新增：

```text
/home/users1/ljian/LFV/scripts/run_hamer_demo_env.sh
```

- 已验证 HaMeR 环境：

```text
/home/users1/ljian/anaconda3/envs/hamer/bin/python
torch 2.5.1+cu121
```
