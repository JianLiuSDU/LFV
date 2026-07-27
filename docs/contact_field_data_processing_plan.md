# Contact Field 数据处理计划与当前修改记录

本文档记录 LFV 仓库中为“从人类 RGB-D 操作视频中提取被操作物体表面的任务相关接触区域”所做的理解、当前修改和后续计划。

当前任务只负责数据处理、标签构造、中间结果保存和可视化检查，不修改后续训练 dataset、模型结构或训练逻辑。

## 1. 重新明确任务目标

项目已经能够从人类 RGB-D 操作视频中提取：

- 被操作物体点云；
- 目标物体点云；
- 被操作物体相对于目标物体的 6D 运动轨迹。

现在需要补充的是“抓哪里”的监督信号，也就是：

- 人手与被操作物体建立接触时；
- 接触对应到被操作物体表面的任务相关区域；
- 例如倒水时的杯把或杯身，开抽屉时的把手；
- 最终用于约束 AnyGrasp / GraspNet 只在任务相关区域附近生成抓取候选。

这里不需要从视频中直接恢复人手的 6D 抓取姿态，也不需要生成新的点云。后续模型的学习目标是每个物体点的接触热力值。

## 2. 为什么仍然需要“锚点物体点云”

这里的“锚点物体点云”不是为了恢复手的姿态，而是为了定义最终标签所在的固定物体表面。

原因如下：

1. 最终数据形式要求是：

```text
接触前无遮挡帧中的被操作物体点云
每个点保留 RGB 像素对应
每个点附加 0 到 1 的接触热力值
```

因此必须先有一个接触前、无遮挡、深度可靠的物体点云，作为逐点标签的载体。

2. 接触发生时，真实接触区域往往被手遮挡：

- 接触帧中的物体 mask 可能缺失；
- 接触帧中的物体深度可能被手覆盖；
- 直接在接触帧用“手 mask 与物体 mask 交集”会漏掉真实接触面。

因此更合理的做法是：

- 在接触前无遮挡帧恢复完整物体表面；
- 在接触窗口中判断手和物体表面哪里接近或接触；
- 把接触证据映射回接触前的物体表面；
- 最终得到固定点云上的逐点热力标签。

3. 这个锚点坐标系也是任务坐标系：

- `points_object = points_camera - object_center`
- 目标物体和已有 6D 轨迹也应使用同一个被操作物体中心；
- 这样后续抓取区域、物体运动和目标物体相对位置在同一坐标系下。

所以，锚点物体点云不是额外要学的对象，也不是手姿态恢复的一部分，而是“接触热力标签的坐标载体”。

## 3. 当前仓库已有可复用能力

### 3.1 RGB-D episode 读取

已有代码：

```text
lfv/data_processing/episode_io.py
```

可复用数据：

```text
episode_x/
  rgb/
  depth/
  camera_0.mp4
  meta.json
  timestamps.npy
```

当前 raw/processed episode 采用 zarr 保存 RGB 和 depth。`prepare` 阶段只建立软链接，不重编码数据。

### 3.2 相机内参与深度尺度

已有代码：

```text
lfv/pipeline/tracking.py
diffusion_policy_3d/dataset/lfv_dataset.py
```

可复用逻辑：

- 从 `meta.json` 读取 `depth_intrinsics_original`；
- 从 `meta.json` 读取 `depth_scale`；
- 将 depth uint16 转成米制深度；
- 进行 2D 像素到 3D camera point 的反投影。

### 3.3 GroundingDINO 物体检测

已有代码：

```text
lfv/pipeline/dino_bbox.py
```

当前用途：

- 在第一帧根据文本 prompt 检测被操作物体和目标物体 bbox。

注意：

- 当前仓库里的 DINO 指的是 GroundingDINO 检测；
- 还没有 DINOv2 稠密语义特征提取。

### 3.4 SAM2 物体分割

已有代码：

```text
lfv/pipeline/sam2_mask.py
```

当前用途：

- 根据 DINO bbox 生成第一帧被操作物体 mask 和目标物体 mask。

可复用结果：

```text
sam_mask/affordance_mask.npy
target_sam_mask/target_mask.npy
```

### 3.5 点采样、点跟踪与物体轨迹

已有代码：

```text
lfv/pipeline/sample_points.py
lfv/pipeline/tracking.py
lfv/pipeline/se3_trajectory.py
```

可复用结果：

```text
sample_points/sampled_2d_uniform.npy
point_tracking/tapip3d_result.npz
se3_trajectory/se3_relative_trajectory.npz
se3_trajectory/dp_action_trajectory.npz
```

其中：

- TAPIP3D 已经能跟踪锚点帧物体表面点；
- `se3_relative_trajectory.npz` 已经保存物体从第 0 帧到每一帧的刚体变换；
- 这些结果可以用于把接触前物体表面投影到接触帧，而不是在接触帧重新依赖被遮挡的物体 mask。

### 3.6 3D 几何处理

当前依赖环境中可用：

- `numpy`
- `scipy`
- `opencv`
- `matplotlib`
- `open3d`

已实现的最小闭环暂时主要使用 `numpy/scipy/opencv/matplotlib`，后续可以进一步接入 Open3D 做更强的点云法向、离群点去除和连通性可视化。

## 4. 当前仍缺失的能力

当前 LFV 仓库内还缺少：

1. 人手检测与精细手 mask；
2. 手物接触置信度检测；
3. 首次稳定接触时刻自动判断；
4. 接触前无遮挡锚点帧自动选择；
5. 逐帧物体 mask 或 SAM2 视频传播；
6. DINOv2 稠密特征提取；
7. 自动批量处理和人工 review 清单。

后续可以参考或接入：

- GroundingDINO：开放词汇手和物体检测；
- SAM2 / Grounded-SAM-2：手和物体精细分割、视频传播；
- 100DOH hand_object_detector：手物接触和被接触物体判断；
- CoTracker：二维点跟踪补充；
- DINOv2：逐点语义特征；
- Open3D：点云、法向、连通性处理和可视化。

## 5. 当前已完成的修改

### 5.1 新增 Contact Field 数据处理模块

新增文件：

```text
lfv/pipeline/contact_field.py
```

当前实现的是最小闭环版本，核心功能包括：

- 读取现有 processed episode；
- 读取 RGB-D zarr；
- 读取相机内参和深度尺度；
- 读取锚点物体 mask；
- 从锚点 RGB-D 构造物体点云；
- 保留每个点的锚点像素坐标；
- 计算中心化米制点云；
- 计算尺度归一化点云；
- 估计点云法向；
- 复用已有 SE(3) 轨迹把锚点物体表面投影到接触帧；
- 根据手部 mask 或手工 bbox 距离变换计算接触证据；
- 可选结合深度一致性；
- 多帧接触证据用 max 聚合；
- 根据接触种子拟合单个加权椭圆高斯热力图；
- 将二维热力值赋给锚点物体点云；
- 使用 3D KNN、法向和主连通分量进行第一版表面修正；
- 保存 npz、json 和可视化。

### 5.2 新增 Contact Field 配置

新增文件：

```text
configs/pipeline/contact_field.yaml
```

集中管理：

- processed 数据目录；
- `anchor_frame`；
- `contact_window`；
- hand bbox / hand mask 输入；
- 输出目录；
- 点云数量；
- 距离变换参数；
- 深度一致性参数；
- 椭圆高斯参数；
- 3D 表面修正参数；
- 质量评价阈值。

### 5.3 接入现有 pipeline

修改文件：

```text
scripts/run_pipeline.py
```

新增 stage：

```text
contact -> lfv.pipeline.contact_field
```

因此可以用现有 pipeline 入口执行：

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/contact_field.yaml \
  --steps contact \
  --episodes 0
```

### 5.4 新增单步执行脚本

新增文件：

```text
scripts/run_step6_contact.py
```

用于单条视频调试，例如：

```bash
python scripts/run_step6_contact.py \
  --config configs/pipeline/contact_field.yaml \
  --episodes 0 \
  --anchor-frame 0 \
  --contact-window 60 90 \
  --hand-bbox 130 120 270 285 \
  --overwrite
```

也可以把 smoke 输出写到仓库内可写目录：

```bash
--output-dir /home/users1/ljian/LFV/data/contact_smoke_episode0
```

### 5.5 新增输出检查脚本

新增文件：

```text
tools/check_contact_field_outputs.py
```

用于检查一条输出是否完整：

- 必需 key 是否存在；
- 点云、像素、热力标签长度是否一致；
- 是否存在 NaN/Inf；
- `contact_heat` 是否在 `[0, 1]`；
- 可视化文件是否生成。

### 5.6 新增核心测试

新增文件：

```text
tests/test_contact_field_core.py
```

测试内容：

- 锚点点云构造是否保留像素对应；
- 二维椭圆热力图是否被物体 mask 正确限制；
- 3D 主连通分量修正是否能删除离散高热异常点。

### 5.7 配置对象写入修复

修改文件：

```text
lfv/utils/config.py
```

新增 `Config.__setattr__`，保证命令行覆盖如 `--episodes`、`--anchor-frame`、`--contact-window` 能真正写入配置对象。

这个修改只影响数据处理脚本参数覆盖，不涉及训练 dataset 或模型逻辑。

## 6. 当前输出格式

每条 episode 默认输出：

```text
episode_x/contact_field/
```

当前最小闭环输出：

```text
contact_field/
  contact_field.npz
  contact_meta.json
  viz/
    viz_anchor_mask.png
    viz_contact_seeds.png
    viz_contact_heatmap.png
    viz_contact_points_3d.png
    viz_reprojection.png
```

`contact_field.npz` 包含：

```text
points_camera
points_object_m
points_object_norm
pixels_uv
normals_camera
contact_evidence
contact_heat_raw
contact_heat
heatmap_2d
object_mask_anchor
seed_mask
object_center_camera
object_scale
anchor_frame
contact_frames
```

`contact_meta.json` 包含：

```text
episode
anchor_frame
contact_frames
point_count
object_mask_pixels
valid_depth_ratio
object_center_camera
object_scale
seed_count
heat_max
heat_mean
heat_area_ratio
evidence
gaussian
surface_correction
dino_features
quality
```

## 7. 已完成验证

核心测试：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spot
python -m unittest tests.test_contact_field_core
```

结果：

```text
Ran 3 tests
OK
```

编译检查：

```bash
python -m py_compile \
  lfv/pipeline/contact_field.py \
  lfv/utils/config.py \
  scripts/run_step6_contact.py \
  tools/check_contact_field_outputs.py
```

结果：通过。

单条真实 episode smoke：

```bash
python scripts/run_step6_contact.py \
  --config configs/pipeline/contact_field.yaml \
  --episodes 0 \
  --anchor-frame 0 \
  --contact-window 60 90 \
  --hand-bbox 130 120 270 285 \
  --output-dir /home/users1/ljian/LFV/data/contact_smoke_episode0 \
  --overwrite
```

输出检查：

```bash
python tools/check_contact_field_outputs.py \
  /media/ljian/lj/data_3d/pickNplace_lfv/episode_0 \
  --output-dir /home/users1/ljian/LFV/data/contact_smoke_episode0
```

检查结果：

```text
point_count: 4096
heat_max: 1.0
heat_mean: 0.459
```

注意：这个 smoke 使用的是粗手工 bbox，只验证数据闭环可以跑通，不代表最终自动标签质量达标。正式版本应使用精细 hand mask 和接触置信度。

## 8. 后续修改计划

后续继续只做数据处理和阶段产物保存，不接入训练逻辑。

### 阶段 1：把最小闭环变成可靠人工/半自动标注工具

目标：先支持对少量视频稳定生成可 review 的 Contact Field 标签。

计划：

1. 支持每条 episode 独立配置：
   - anchor frame；
   - contact window；
   - 每帧 hand bbox；
   - 每帧 hand mask；
   - 每帧 contact confidence。

2. 明确输入来源：
   - `hand_mask` 来源允许 `good`；
   - 粗 `hand_bbox` 来源默认标记为 `review`；
   - 无接触置信度时标记为 `review`。

3. 强化可视化：
   - 锚点图像 + 物体 mask；
   - 接触窗口中 hand mask / bbox 与物体投影；
   - 接触种子权重；
   - 二维椭圆热力图；
   - 三维点云热力图；
   - 被 3D 连通性删除的异常高热点。

### 阶段 2：自动手部检测与分割

目标：替代手工 bbox。

计划：

1. 复用现有 GroundingDINO 代码检测 hand bbox：

```text
prompt: "hand ."
```

2. 复用 SAM2 生成精细 hand mask。

3. 保存阶段结果：

```text
hand_detection/
  frame_000060_bbox.npy
  frame_000060_score.json

hand_mask/
  frame_000060.npy
  frame_000060_overlay.png
```

4. Contact Field 阶段优先读取 `hand_mask/`，再回退到 bbox。

### 阶段 3：接入手物接触置信度

目标：不只看空间接近，还判断是否真的建立接触。

计划：

1. 接入 100DOH hand_object_detector 或等价结果。
2. 对每帧保存：

```text
contact_detection/
  contact_scores.npy
  contacted_object_scores.npy
  contact_detection.json
```

3. 在 Contact Field 中使用：

```text
contact_evidence_i_frame =
  distance_weight *
  contact_confidence *
  optional_depth_consistency
```

### 阶段 4：自动首次稳定接触窗口

目标：从连续多帧判断接触开始。

计划：

1. 计算每帧 hand-object proximity。
2. 结合 contact confidence。
3. 使用连续帧规则：
   - 首次超过阈值；
   - 持续 N 帧；
   - 只取接触建立后的短窗口。

保存：

```text
contact_timing/
  contact_scores.npy
  contact_timing.json
  contact_scores_plot.png
```

### 阶段 5：自动选择锚点帧

目标：选择接触前最近、无遮挡、物体完整、深度可靠的帧。

计划：

1. 从首次稳定接触帧向前搜索。
2. 对候选帧评分：
   - 物体 mask 面积；
   - 有效深度比例；
   - hand-object 遮挡比例；
   - 物体运动幅度。

保存：

```text
anchor_selection/
  anchor_candidates.json
  anchor_frame_overlay.png
```

### 阶段 6：DINOv2 逐点语义特征

目标：为物体点附加语义条件，但仍只保存数据。

计划：

1. 对锚点 RGB 提取稠密 DINOv2 feature map。
2. 根据 `pixels_uv` 采样每个点的特征。
3. 保存：

```text
contact_field.npz -> dino_features
```

或单独保存：

```text
dino_features.npy
```

### 阶段 7：批量处理与 manifest

目标：为后续人工筛选和训练前数据准备提供清单。

计划：

```text
contact_field_manifest.json
contact_field_summary.csv
```

每条记录包括：

- episode；
- anchor frame；
- contact window；
- hand mask 来源；
- contact confidence 来源；
- heat statistics；
- quality；
- review reason；
- 输出路径。

## 9. 重要边界

后续继续保持：

- 不恢复人手 6D 抓取姿态；
- 不生成新的点云；
- 不修改训练 dataset；
- 不修改模型和训练逻辑；
- 只从视频和已有中间结果中构造、保存、检查和可视化接触热力标签；
- 最终输出是固定物体点云上的逐点接触热力值。
