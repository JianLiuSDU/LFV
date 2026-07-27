# Hand Pouring Contact And Grasp Handoff

## 任务背景

项目目标是从人类 RGB-D 操作视频中学习机器人操作。当前处理对象是倒水任务：

```text
raw data:       /media/ljian/lj/hand_data/pouring
processed data: /media/ljian/lj/data_3d/hand_pouring_lfv
config:         /home/users1/ljian/LFV/configs/pipeline/hand_pouring.yaml
```

现有 LFV 流程已经能够从视频中提取被操作物体和目标物体的点云，并恢复被操作物体相对目标物体的 6D 运动轨迹。当前新增工作的目标不是修改训练 dataset，也不是让模型生成点云，而是补充两类数据标签：

1. 被操作物体表面逐点接触热力标签，也就是“人抓哪里”。
2. 基于 HaMeR thumb-index 关键点的平行夹爪伪标签，用于验证能否从人手接触点构造机器人抓取先验。

所有数据输出都放在 `/media/ljian/lj` 下，避免污染代码仓库。

## 设计共识

### 接触热力标签

接触热力标签的几何对象是接触前无遮挡锚点帧中的被操作物体点云。每个点保留其锚点 RGB 像素，并附加 `contact_heat in [0, 1]`。

关键原则：

- 不直接用手掩码和物体掩码求交，因为接触时表面容易被手遮挡。
- 使用接触窗口中手掩码距离变换作为接触证据。
- 通过锚点物体点云和已有 SE(3) 轨迹把接触证据对齐回锚点物体表面。
- 用加权椭圆高斯拟合 2D 热力图，再赋值到点云。
- 用 KNN、法向和连通性做三维表面修正。

### HaMeR Thumb-Index 抓取伪标签

最新共识是：第一版不使用掌心、手腕或 MCP 点构造完整 6D 姿态。HaMeR 主要用于提供手部 2D 关键点，尤其是：

```text
4: thumb tip
8: index tip
```

thumb/index 用来确定：

- 两个表面接触点；
- 夹爪中心；
- 夹爪闭合方向；
- 夹爪宽度。

接近方向 `approach` 不应由掌心默认决定。当前初版使用物体表面法向 `-normal`，但已经发现对单视角杯把点云会偏向相机深度方向。后续应改成多 approach 候选或 top-down/task-constrained 先验。

## 已经完成的代码

### 数据处理 pipeline 模块

已新增或扩展：

```text
lfv/pipeline/hand_bbox.py
lfv/pipeline/hand_mask.py
lfv/pipeline/hand_segmentation.py
lfv/pipeline/contact_timing.py
lfv/pipeline/contact_heatmap.py
lfv/pipeline/contact_field.py
lfv/pipeline/dinov2_features.py
```

`scripts/run_pipeline.py` 已加入以下 stage：

```text
hand_bbox
hand_mask
timing
dinov2
contact_heatmap
```

批处理/检查脚本：

```text
scripts/run_hand_pouring_contact_batch.sh
tools/check_hand_pouring_contact_batch.py
tools/check_contact_field_outputs.py
```

### HaMeR thumb-index grasp pipeline stages（已并入主流程）

新增模块：

```text
lfv/pipeline/hamer_hand_pose.py          # stage: hamer
lfv/pipeline/thumb_index_grasp_label.py  # stage: thumb_index_grasp
```

`scripts/run_pipeline.py` 新增 stage：

```text
hamer
thumb_index_grasp
```

`configs/pipeline/hand_pouring.yaml` 新增 `hamer:` 与 `thumb_index_grasp:` 配置段。

批处理/检查脚本：

```text
scripts/run_hand_pouring_grasp_batch.sh
tools/check_hand_pouring_grasp_batch.py
```

设计要点：

- `hamer` stage 把所有缺失关键点的帧汇总到 `<processed_root>/_hamer_batch_staging/`，用一次 HaMeR demo 跑完全部 episode，再按 `<episode>__frame_XXXXXX` 文件名前缀分发回各 episode 的 `hamer_output/skeleton2d/`。
- 通过解析 demo 日志中的 `Processing image:` 行记录 `frames_attempted` 到各 episode 的 `hamer_output/hamer_run_meta.json`，区分"已检测但无人/无手"与"尚未处理"，崩溃后可断点续跑（demo 对无人图像不写任何输出，不能只靠输出文件判断）。
- `thumb_index_grasp` 在无任何有效候选时不崩溃，写 `valid=False` 的 npz 和 `quality=reject` 的 meta（遵循"标记 good/review/reject，不删除失败样本"）。
- HaMeR 计算需在空闲 GPU 上运行（本机 GPU 0 常被其他任务占用导致 Detectron2 OOM，批处理用 `CUDA_VISIBLE_DEVICES=1`）。

运行命令：

```bash
cd /home/users1/ljian/LFV
CUDA_VISIBLE_DEVICES=1 bash scripts/run_hand_pouring_grasp_batch.sh
```

### GraspNet ROI 验证

新增：

```text
tools/verify_episode0_graspnet_contact_roi.py
scripts/visualize_episode0_graspnet_contact_roi.sh
```

作用：

- 读取 `contact_heatmap`；
- 生成 contact ROI mask；
- 可调用 GraspNet API；
- 若 GraspNet API 未启动，则用 contact heat fallback 生成小夹爪可视化；
- 输出 2D/3D 可视化。

episode_0 输出目录：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/graspnet_contact_roi_verify
```

重要输出：

```text
contact_heat_overlay.png
contact_roi_mask_overlay.png
selected_grasp_overlay.png
selected_grasp_3d.png
graspnet_contact_filter_report.json
```

### HaMeR 接入

已有 HaMeR 仓库：

```text
/home/users1/ljian/hamer
```

已用符号链接接入 LFV third party：

```text
/home/users1/ljian/LFV/third_party/hamer -> /home/users1/ljian/hamer
```

HaMeR 环境：

```text
/home/users1/ljian/anaconda3/envs/hamer/bin/python
```

已验证：

```text
torch 2.5.1+cu121
MANO_RIGHT.pkl exists at /home/users1/ljian/hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

运行 wrapper：

```text
scripts/run_hamer_demo_env.sh
```

该脚本固定使用 HaMeR 自己的 conda 环境，避免污染 LFV 当前环境。

### HaMeR Thumb-Index 抓取伪标签

新增：

```text
tools/process_episode0_hamer_thumb_index_grasp.py
tools/visualize_episode0_hamer_thumb_index_grasp_open3d.py
docs/hamer_grasp_pseudo_label_plan.md
```

处理脚本会：

1. 从 `contact_timing` 中选接触窗口；
2. 导出窗口 RGB 帧到 `hamer_input`；
3. 调用 HaMeR demo；
4. 读取 `hamer_output/skeleton2d/*_hand_2d.npy`；
5. 选择和手掩码/接触热力最一致的手；
6. 取 thumb/index 的原图 2D 像素；
7. 用 D455 深度局部中位数反投影到相机米制坐标；
8. 查最近物体表面点 `q_thumb`, `q_index`；
9. 由两个表面点构造平行夹爪候选；
10. 从窗口候选中选择 median-like 代表姿态；
11. 保存 `npz`, `json`, 2D 图和 3D 图。

运行命令：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/process_episode0_hamer_thumb_index_grasp.py --overwrite
```

Open3D 查看命令：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/visualize_episode0_hamer_thumb_index_grasp_open3d.py
```

## episode_0 已完成输出

### 接触时刻

文件：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/contact_timing/contact_timing.json
```

结果：

```text
anchor_frame: 39
contact_start: 45
contact_end: 66
contact_frames: [45, 48, 51, 54, 57, 60, 63, 66]
quality: good
```

### 接触热力图

文件：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/contact_heatmap/contact_heatmap.npz
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/contact_heatmap/contact_heatmap_meta.json
```

核心字段：

```text
points_camera: (4096, 3)
points_object_m: (4096, 3)
points_object_norm: (4096, 3)
normals_camera: (4096, 3)
pixels_uv: (4096, 2)
object_center_camera: (3,)
object_scale: scalar
heatmap_2d: (480, 640)
contact_heat: (4096,)
contact_frames: (4,)
```

结果观察：

- 热区落在杯把附近；
- 可视化在 `contact_heatmap/viz`。

### DINOv2 特征

文件：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/dinov2_features/point_dinov2_features.npy
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/dinov2_features/anchor_dinov2_grid.npz
```

已完成点级 DINO 特征附着，点数与 contact heat 的点云对应。

### HaMeR Thumb-Index 抓取伪标签

输出目录：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0/hamer_grasp_pseudo_label
```

重要文件：

```text
grasp_pseudo_label.npz
grasp_pseudo_label_meta.json
selected_grasp_graspnet_row.npy
viz/selected_grasp_overlay_2d.png
viz/window_candidates_2d.png
viz/selected_grasp_3d.png
hamer_input/
hamer_output/
```

本次结果：

```text
frames_requested: [45, 48, 51, 54]
frames_valid: [48, 51, 54]
selected_frame: 51
selected_hand_id: person0_left_hand_2d
quality: good
confidence: 0.5279943011246456
width_m: 0.024893260456621647
mean_finger_surface_dist_m: 0.020446787761393675
```

`grasp_pseudo_label.npz` 字段：

```text
T_grasp_cam: (4, 4)
T_grasp_object: (4, 4)
rotation_6d: (6,)
translation_object: (3,)
width_m: scalar
q_thumb_object: (3,)
q_index_object: (3,)
q_thumb_cam: (3,)
q_index_cam: (3,)
candidate_T_cam: (3, 4, 4)
candidate_T_object: (3, 4, 4)
candidate_width_m: (3,)
candidate_frames: (3,)
valid: bool
confidence: scalar
selected_frame: scalar
```

## 左右手处理细节

HaMeR 使用 `MANO_RIGHT.pkl`。左手不是用左手 MANO 直接预测，而是：

1. ViTPose 检出 left hand 和 right hand 2D keypoints。
2. 如果是左手，`ViTDetDataset` 将 hand crop 水平翻转。
3. 翻转后的左手送入右手 MANO 模型。
4. 输出 mesh/vertices 时根据 `right` 标志镜像回来。

相关代码：

```text
/home/users1/ljian/hamer/hamer/datasets/vitdet_dataset.py
/home/users1/ljian/hamer/demo.py
```

这次 grasp 伪标签实际使用的是：

```text
hamer_output/skeleton2d/frame_*_person*_left_hand_2d.npy
hamer_output/skeleton2d/frame_*_person*_right_hand_2d.npy
```

这些是 ViTPose 原图坐标系下的 2D hand keypoints。当前没有直接使用 `hamer_keypoints/*_pred_keypoints_2d.npy`，因为 HaMeR 保存的 `pred_keypoints_2d` 是 crop-normalized 坐标，不能直接当原图像素使用。

## 当前抓取构造方式

在 `tools/process_episode0_hamer_thumb_index_grasp.py` 中：

```text
q_thumb = nearest_object_surface(thumb_tip_3d)
q_index = nearest_object_surface(index_tip_3d)
center = 0.5 * (q_thumb + q_index)
closing = normalize(q_index - q_thumb)
width = norm(q_index - q_thumb) + width_margin
```

当前 approach 是：

```text
normal = normal(q_thumb) + normal(q_index)
approach = normalize(-normal)
approach = orthogonalize(approach, closing)
binormal = cross(approach, closing)
R = [approach, closing, binormal]
tcp = center - approach * tcp_to_contact_offset
```

坐标约定：

```text
X = approach
Y = closing
Z = binormal
```

episode_0 当前实际选中姿态：

```text
approach = [0.003, 0.156, 0.988]
closing  = [0.077, -0.985, 0.155]
tcp      = [-0.288269, -0.028562, 0.663550]
```

OpenCV 相机坐标中 `z` 是从相机指向场景深处。因此当前 `approach.z = 0.988`，确实偏向相机深度方向。这是目前最重要的待改进点。

## 已知问题

### 1. approach 方向仍不理想

用户希望更接近“从上往下”的抓取方向。当前用 `-surface_normal`，在单视角杯把点云上容易被相机视线方向影响，导致 approach 看起来从摄像头方向往深处延伸。

后续建议：

- 保留 `center/closing/width` 来自 thumb-index；
- 不再单独依赖 `surface_normal`；
- 增加 approach candidates：
  - top-down；
  - camera-to-object；
  - surface-normal；
  - 围绕 closing 旋转的一组候选；
- 用碰撞、接触点深度、接触热力和机器人可达性评分选择 approach。

### 2. 需要判断 TCP 往里抓了多少

当前：

```text
tcp = center - approach * tcp_to_contact_offset
tcp_to_contact_offset = 0.045 m
```

理论上接触中心在夹爪坐标系中的 approach 深度为：

```text
depth_into_gripper = dot(center - tcp, approach) = 0.045 m
```

但是否能抓住杯把，不能只看 offset。应把物体点云变换到夹爪坐标系：

```text
p_grasp = R^T * (p_cam - tcp)
```

并检查：

```text
x: 沿 approach 方向的进入深度
y: closing 方向，两指之间
z: binormal 方向，指宽范围
```

需要输出诊断指标：

```text
q_thumb_grasp, q_index_grasp
contact_depth_x
high_heat_points_inside_gripper_ratio
points_between_fingers_count
finger_collision_count
palm_collision_count
```

### 3. HaMeR demo 依赖人体检测

HaMeR 官方 demo 先用 Detectron2 检人，再用 ViTPose 找手。如果视频只露出局部手臂、人体检测失败，HaMeR 可能无输出。

episode_0 这次能跑通，但批处理前建议新增“外部 hand bbox 推理入口”，直接复用已有 `hand_bbox/frame_*.npy`，避免依赖全身 person detection。

### 4. HaMeR 原始 3D 没有作为米制坐标使用

这是有意为之。HaMeR 是单目 hand mesh recovery，不能直接相信其全局深度和尺度。当前只用 ViTPose 原图 2D keypoints，并用 D455 深度反投影。

### 5. 尚未接入训练 dataset

按用户要求，当前没有修改后续训练 dataset 逻辑。所有新增结果只是数据处理输出。

## 下一步建议

### Step 1: 增加抓取诊断脚本

新增一个只读脚本，例如：

```text
tools/diagnose_episode0_thumb_index_grasp_geometry.py
```

功能：

- 读取 `grasp_pseudo_label.npz` 和 `contact_heatmap.npz`；
- 把点云转到夹爪坐标系；
- 统计接触点和高热物体点相对夹爪的 x/y/z 范围；
- 输出 `depth_into_gripper`；
- 可视化夹爪坐标盒；
- 判断当前 TCP offset 是否能覆盖杯把。

### Step 2: 改 approach 生成

把当前单一：

```text
approach = -surface_normal
```

改为候选评分：

```text
approach_candidates = [
  top_down,
  camera_to_object,
  surface_normal,
  rotated_about_closing(...)
]
```

先实现 `top_down` 或 `top_down_with_small_tilt`。注意需要确定世界/桌面 z 方向；如果只有相机坐标，可以先通过桌面平面拟合得到 tabletop normal。

### Step 3: 外部 hand bbox HaMeR 入口

给 HaMeR 增加或单独写一个脚本：

```text
third_party/hamer/run_lfv_hand_bbox.py
```

输入：

```text
RGB frame
hand bbox
is_right guess or both-hand mode
```

输出：

```text
原图坐标系下 2D hand keypoints
可选 MANO mesh/keypoints
```

这样批处理时不依赖 person detector。

### Step 4: 批处理多个 episode（已完成）

已抽象为 `lfv/pipeline/hamer_hand_pose.py` 和 `lfv/pipeline/thumb_index_grasp_label.py`，并加入 `scripts/run_pipeline.py`、`scripts/run_hand_pouring_grasp_batch.sh`、`tools/check_hand_pouring_grasp_batch.py`。episode_0 回归验证：pipeline 输出与原 `tools/process_episode0_hamer_thumb_index_grasp.py` 逐字段一致。

2026-07 全量批处理结果（149 episodes）：

```text
quality: good 141, review 7, missing 1
confidence min/median/max: 0.089/0.426/0.727
width_m min/median/max: 0.015/0.021/0.035
finger_surface_dist min/median/max: 0.005/0.021/0.032
```

- review episodes: 3, 7, 14, 86, 95, 142, 143（有效候选帧不足或置信度低，保留待查，未删除）
- 唯一失败是 episode_61：`contact_timing` 无 contact_start/contact_frames（上游接触检测问题，非本步骤 bug）

### Step 5: 与 GraspNet/AnyGrasp 对接

后续 GraspNet 限制策略：

- 输入仍可用 full object mask 或 contact ROI mask；
- 候选按 contact heat、thumb-index center、closing 方向、width 和碰撞进行重排序；
- `selected_grasp_graspnet_row.npy` 当前已经保存为 GraspNet row 风格，可作为接口参考。

## 常用命令

全量批处理 thumb-index grasp（hamer + thumb_index_grasp 两个 stage）：

```bash
cd /home/users1/ljian/LFV
CUDA_VISIBLE_DEVICES=1 bash scripts/run_hand_pouring_grasp_batch.sh
```

只跑部分 episode / 重算：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_hand_pouring_grasp_batch.sh --episodes 3 7 14
CUDA_VISIBLE_DEVICES=1 bash scripts/run_hand_pouring_grasp_batch.sh --overwrite
```

查看批处理质量汇总：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  tools/check_hand_pouring_grasp_batch.py --show-bad
```

检查 episode_0 contact batch：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/check_hand_pouring_contact_batch.py
```

重新生成 episode_0 HaMeR thumb-index grasp：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/process_episode0_hamer_thumb_index_grasp.py --overwrite
```

Open3D 查看任意 episode 的 HaMeR thumb-index grasp（推荐）：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/visualize_hamer_thumb_index_grasp_open3d.py \
  --episode-dir /media/ljian/lj/data_3d/hand_pouring_lfv/episode_0 \
  --show-candidates
```

参数：

- `--object-frame`：切换到物体系（相机轴向、点云质心为原点），便于跨 episode 比较
- `--show-candidates`：同时显示窗口内所有候选位姿（灰色）
- `--heat-field contact_heat_raw`：看未做 3D 表面修正的原始热力

旧版 episode_0 专用 viewer：

```bash
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/visualize_episode0_hamer_thumb_index_grasp_open3d.py
```

Open3D 查看 GraspNet/contact ROI 验证：

```bash
cd /home/users1/ljian/LFV
bash scripts/visualize_episode0_graspnet_contact_roi.sh
```

如果要真实调用 GraspNet API，需要先启动：

```bash
cd /home/users1/ljian/graspnet-baseline
/home/users1/ljian/anaconda3/envs/graspnet/bin/python graspnet_api_server.py
```

然后：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/im2flow2act/bin/python \
  tools/verify_episode0_graspnet_contact_roi.py --call-api
```

## 数据存储格式与训练提取方法

所有新增产出都在 `<processed_root>/<episode>/` 下，训练代码读取时应以这些路径为准。

### 接触热力图

位置：

```text
<episode>/contact_heatmap/contact_heatmap.npz
<episode>/contact_heatmap/contact_heatmap_meta.json
```

`contact_heatmap.npz` 主要字段：

```text
points_camera            (4096, 3)  f32   # anchor 帧相机系点云（米）
points_object_m          (4096, 3)  f32   # 相机轴向、点云质心为原点的物体系
points_object_norm       (4096, 3)  f32   # points_object_m / object_scale
normals_camera           (4096, 3)  f32   # 相机系法向
pixels_uv                (4096, 2)  i32   # 每点对应 anchor 帧像素
object_center_camera     (3,)       f32   # 点云质心
object_scale             scalar     f32   # 归一化尺度
heatmap_2d               (480, 640) f32   # anchor 图平面热力图
contact_heat             (4096,)    f32   # 逐点接触热力（3D 表面修正后，训练主用）
contact_heat_raw         (4096,)    f32   # 2D 热力直接投影（未修正）
anchor_object_mask       (480, 640) u8    # anchor 帧物体掩码
contact_frames           (T,)       i32   # 实际使用的接触帧
```

Python 读取示例：

```python
import numpy as np
ep = "/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0"
heat = np.load(f"{ep}/contact_heatmap/contact_heatmap.npz")
points_obj = heat["points_object_m"]       # (4096, 3)
contact_heat = heat["contact_heat"]        # (4096,)
```

注意：DINOv2 点特征是 512 维，附着在 `sample_points/sampled_2d_uniform.npy` 的 512 个采样点上（`dinov2_features/point_dinov2_features.npy` 形状为 `(512, 384)`），**与 `contact_heat` 的 4096 点云不是逐点对应**。训练若要把 512 特征贴到 4096 点云上，需要通过 `pixels_uv` / `point_pixels_uv.npy` 做像素最近邻映射，或用 `anchor_dinov2_grid.npz` 对 4096 个像素位置重新采样。

### HaMeR thumb-index 抓取伪标签

位置：

```text
<episode>/hamer_grasp_pseudo_label/grasp_pseudo_label.npz
<episode>/hamer_grasp_pseudo_label/grasp_pseudo_label_meta.json
<episode>/hamer_grasp_pseudo_label/selected_grasp_graspnet_row.npy
<episode>/hamer_grasp_pseudo_label/hamer_output/skeleton2d/
```

`grasp_pseudo_label.npz` 主要字段：

```text
T_grasp_cam        (4, 4)  f32   # TCP 在相机系。列约定：X=approach, Y=closing, Z=binormal
T_grasp_object     (4, 4)  f32   # 同旋转；平移 = tcp - object_center_camera（与 points_object_m 同系）
rotation_6d        (6,)    f32   # T_grasp_object 前两列（approach, closing）列优先展开
translation_object (3,)    f32   # TCP 在物体系位置
width_m            scalar  f32   # 夹爪开口 = ||q_index - q_thumb|| + 0.012
q_thumb_object     (3,)    f32   # 拇指接触点（物体系）
q_index_object     (3,)    f32   # 食指接触点（物体系）
q_thumb_cam        (3,)    f32   # 拇指接触点（相机系）
q_index_cam        (3,)    f32   # 食指接触点（相机系）
candidate_T_cam    (N,4,4) f32   # 窗口内所有候选（相机系）
candidate_T_object (N,4,4) f32   # 窗口内所有候选（物体系）
candidate_width_m  (N,)    f32   # 候选宽度
candidate_frames   (N,)    i32   # 候选帧号
valid              bool          # quality != reject
confidence         scalar  f32   # 综合置信度
selected_frame     scalar  i32   # 代表帧；reject 时为 -1
```

Python 读取与过滤示例：

```python
import json, numpy as np
ep = "/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0"
grasp = np.load(f"{ep}/hamer_grasp_pseudo_label/grasp_pseudo_label.npz")
meta = json.load(open(f"{ep}/hamer_grasp_pseudo_label/grasp_pseudo_label_meta.json"))

if not bool(grasp["valid"]) or meta["quality"] == "reject":
    # 该 episode 不可用
    pass

T_obj = grasp["T_grasp_object"]         # (4, 4)
width = float(grasp["width_m"])
conf = float(grasp["confidence"])
thumb_obj = grasp["q_thumb_object"]     # (3,)
index_obj = grasp["q_index_object"]     # (3,)

# TCP 与接触中心关系校验
center_obj = 0.5 * (thumb_obj + index_obj)
approach = T_obj[:3, 0]
tcp_obj = T_obj[:3, 3]
depth_into_gripper = np.dot(center_obj - tcp_obj, approach)
print(depth_into_gripper)  # 应 ≈ 0.045
```

**坐标约定强调**：`T_grasp_object` 和 `points_object_m` 不是规范 CAD 物体系，而是"相机轴向 + 点云质心为原点"的平移系。后续训练若需要统一的规范物体坐标，需自行在任务内做配准或零样本对齐。

**reject 约定**：无有效候选时 `valid=False`，位姿字段全为 NaN，`selected_frame=-1`，`confidence=0`。训练代码务必先用 `valid` 或 `meta["quality"]` 过滤。

### 2D 手部关键点（可选）

```text
<episode>/hamer_grasp_pseudo_label/hamer_output/skeleton2d/frame_XXXXXX_personN_left_hand_2d.npy
```

形状 `(21, 3)`，字段 `(u, v, confidence)`，为**原图像素坐标**，可配 D455 深度反投影。
注意 `hamer_output/hamer_keypoints/*_pred_keypoints_2d.npy`（若存在）是 crop 归一化坐标，不能直接反投影。

## 给后续模型的注意事项

- 不要把输出数据写进 LFV 代码目录；数据都应放在 `/media/ljian/lj`。
- 不要修改训练 dataset，除非用户明确要求。
- 不要把 `/home/users1/ljian/hamer` 复制到 LFV；当前用 symlink 即可。
- HaMeR 相关运行必须用 `/home/users1/ljian/anaconda3/envs/hamer/bin/python` 或 `scripts/run_hamer_demo_env.sh`。
- `hamer_keypoints/*_pred_keypoints_2d.npy` 不是原图坐标，不能直接用于 D455 深度反投影。
- 当前 approach 为 `-surface_normal` 近似 top-down；后续若需改成多候选评分，改动会触发 grasp 链重跑。
