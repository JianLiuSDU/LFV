# 新任务数据处理 Runbook

本文档整理从零开始处理一批新任务 RGB-D 操作视频的完整步骤，以 hand_pouring 为已验证样例。
目标产出：物体点云 + 逐点接触热力标签 + HaMeR thumb-index 平行夹爪抓取伪标签。

## 0. 前置条件

### 原始数据格式

原始数据根目录下每个 episode 一个目录，需包含：

```text
episode_k/
├── rgb/            # zarr，RGB uint8（或 camera_0/rgb）
├── depth/          # zarr，uint16 raw depth（或 camera_0/depth），已与 color 对齐
├── meta.json       # D455 内参（color_intrinsics / depth_intrinsics_original）+ depth_scale
├── camera_0.mp4    # 可选，仅作可视化备份
└── timestamps.npy  # 可选
```

`prepare` stage 只建符号链接，不复制数据。

### 环境

| 用途 | python |
|---|---|
| 大部分 stage（dino/timing/dinov2/contact_heatmap/hamer/thumb_index_grasp） | `/home/users1/ljian/anaconda3/envs/tapip3d/bin/python` |
| sam2 / hand_mask | `/home/users1/ljian/anaconda3/envs/sam2/bin/python` |
| HaMeR demo（由 hamer stage 内部 subprocess 调用） | `/home/users1/ljian/anaconda3/envs/hamer/bin/python`（`scripts/run_hamer_demo_env.sh` 固定） |

依赖资源：

- SAM2 checkpoint：`sam2.checkpoint` 指向本机 `sam2.1_hiera_large.pt`
- GroundingDINO：首次运行需联网（配 `HF_ENDPOINT=https://hf-mirror.com`）
- DINOv2 权重：`dinov2.local_weight_path`
- HaMeR：`third_party/hamer` 符号链接 + `_DATA/data/mano/MANO_RIGHT.pkl`
- TAPIP3D checkpoint（仅当需要 track/se3 时）

### 硬性规则

- 所有数据输出写到 `paths.processed_root`（放在 `/media/ljian/lj` 数据盘），不要写进代码仓库。
- 不要修改训练 dataset 逻辑，除非用户明确要求。
- HaMeR 一律通过 `scripts/run_hamer_demo_env.sh` 调用，不污染 LFV 当前环境。

## 1. 创建新任务 yaml

复制 `configs/pipeline/hand_pouring.yaml` 为 `configs/pipeline/<task>.yaml`，需要改的字段：

```yaml
task_name: <task>

paths:
  raw_root: /media/ljian/lj/hand_data/<task>          # 原始数据
  processed_root: /media/ljian/lj/data_3d/<task>_lfv  # 处理输出（自动创建）

object:
  name: "target object ."      # GroundingDINO 总提示（向后兼容字段）
objects:
  affordance:
    prompt: "mug ."            # 被操作物体 ← 必改
  target:
    prompt: "bowl ."           # 目标物体 ← 必改

hand:
  prompt: "hand ."             # 一般不变；检测困难时调 fallback_prompts / box_threshold

contact_timing:                # 接触判定阈值，按任务接触特性调
  contact_distance_px: 8.0
  contact_overlap_ratio: 0.003

contact_heatmap:
  frame_offsets: [-3, 0, 3, 6] # 接触窗口采样（相对 contact_start）
  num_frames: 4

hamer:                         # 一般不变
thumb_index_grasp:             # 一般不变；window_size 需与 hamer.window_size 一致
```

`sam2`、`tapip3d`、`se3`、`dinov2` 段通常直接沿用。

## 2. 处理链条（按依赖顺序）

所有命令在 `/home/users1/ljian/LFV` 下执行，`PY` 用对应环境的 python。

### Step 1: prepare — 建立 processed 目录与符号链接

```bash
tapip3d/bin/python scripts/run_pipeline.py --config configs/pipeline/<task>.yaml --steps prepare
```

输出：`<processed_root>/episode_*/{rgb,depth,camera_0.mp4,meta.json,timestamps.npy}` 符号链接。
可加 `--episodes 0 1 2` 先小批量验证。

### Step 2: dino — GroundingDINO 物体检测

```bash
HF_ENDPOINT=https://hf-mirror.com tapip3d/bin/python \
  scripts/run_pipeline.py --config configs/pipeline/<task>.yaml --steps dino
```

输出：`bbox/affordance_bbox.npy`、`target_bbox/target_bbox.npy`。
检查：抽几个 episode 看 `viz/` 下检测框是否正确；检不出就调 `objects.*.prompt` 或阈值。

### Step 3: sam2 — 物体掩码

```bash
sam2/bin/python scripts/run_pipeline.py --config configs/pipeline/<task>.yaml --steps sam2
```

输出：`sam_mask/affordance_mask.npy`、`target_sam_mask/target_mask.npy`。

### Step 4: sample — 物体表面 2D 采样点

```bash
tapip3d/bin/python scripts/run_pipeline.py --config configs/pipeline/<task>.yaml --steps sample
```

输出：`sample_points/sampled_2d_uniform.npy`（512 点，`{"query_points_2d": (512,2)}`）。

### Step 5（可选）: track + se3 — 物体 6D 轨迹

```bash
tapip3d/bin/python scripts/run_pipeline.py --config configs/pipeline/<task>.yaml --steps track se3
```

输出：`se3_trajectory/se3_relative_trajectory.npz`。
**说明**：hand_pouring 没有跑这一步，`contact_heatmap` 自动回退为 identity 对齐（meta 里 `transform_source: identity`）。
接触窗口内物体基本静止的任务可以跳过；若物体在接触窗口内有明显位移则建议跑。

### Step 6: 接触链 — hand_bbox → hand_mask → timing → dinov2 → contact_heatmap

```bash
CONFIG=configs/pipeline/<task>.yaml bash scripts/run_hand_pouring_contact_batch.sh
```

脚本内部按环境分工（hand_bbox/timing/dinov2/contact_heatmap 用 tapip3d env，hand_mask 用 sam2 env），
支持 `--episodes`、`--overwrite`、`--no-check`。

输出（每 episode）：

```text
hand_bbox/frame_*.npy          # 手检测框（stride=3）
hand_mask/frame_*.npy          # SAM2 手掩码
hand_contact/                  # 手-物接触判定中间量
contact_timing/contact_timing.json
dinov2_features/point_dinov2_features.npy + anchor_dinov2_grid.npz + point_pixels_uv.npy
contact_heatmap/contact_heatmap.npz + contact_heatmap_meta.json + viz/
```

检查：

```bash
tapip3d/bin/python tools/check_hand_pouring_contact_batch.py --config configs/pipeline/<task>.yaml --show-bad
```

失败列表写在 `<processed_root>/contact_heatmap_failed_logs.txt`。
**先确认热区确实落在抓取部位、timing 质量大多为 good，再进行下一步**——抓取标签依赖这一步的全部产出。

### Step 7: 抓取链 — hamer + thumb_index_grasp

```bash
CONFIG=configs/pipeline/<task>.yaml CUDA_VISIBLE_DEVICES=1 \
  bash scripts/run_hand_pouring_grasp_batch.sh
```

说明：

- `hamer` stage 把所有 episode 缺失关键点的帧汇总到 `<processed_root>/_hamer_batch_staging/`，
  一次 HaMeR demo 跑完再分发回各 episode；崩溃可断点续跑（按 demo 日志记录 attempted）。
- `CUDA_VISIBLE_DEVICES` 选空闲 GPU（本机 GPU 0 常被占用，Detectron2 对显存敏感）。
- 支持 `--episodes`、`--overwrite`、`--no-check`。

检查：

```bash
tapip3d/bin/python tools/check_hand_pouring_grasp_batch.py --config configs/pipeline/<task>.yaml --show-bad
```

每 episode 抽查 `hamer_grasp_pseudo_label/viz/selected_grasp_overlay_2d.png` 和 `selected_grasp_3d.png`：
thumb/index 是否落在真实抓取位置、夹爪是否罩住接触区域。
review/reject 的 episode 不会被删除，排查后可 `--episodes k --overwrite` 重算。

## 3. 数据存储位置与格式（训练用）

以 `<processed_root>/<episode>/` 为根。坐标约定：相机系为 OpenCV 约定（z 指向场景深处）；
"物体系" = 相机轴向 + 点云质心为原点的平移系（**不是 CAD 规范系**，见下文注意）。

### 3.1 接触时刻 `contact_timing/contact_timing.json`

```text
anchor_frame: int        # 接触前无遮挡锚点帧
contact_start/end: int
contact_frames: [int]    # 接触窗口帧（stride=3）
quality: good/review/...
```

### 3.2 接触热力标签 `contact_heatmap/contact_heatmap.npz`

| 字段 | shape | 含义 |
|---|---|---|
| `points_camera` | (4096,3) f32 | anchor 帧相机系点云，米 |
| `points_object_m` | (4096,3) f32 | 同上减去点云质心（米制物体系） |
| `points_object_norm` | (4096,3) f32 | `points_object_m / object_scale`（scale=半径 95 分位数） |
| `normals_camera` | (4096,3) f32 | 相机系法向 |
| `pixels_uv` | (4096,2) i32 | 每点对应的 anchor 帧像素 |
| `object_center_camera` | (3,) f32 | 点云质心（物体系原点） |
| `object_scale` | 标量 f32 | 归一化尺度 |
| **`contact_heat`** | **(4096,) f32 [0,1]** | **逐点接触热力标签（3D 表面修正后，训练主用）** |
| `contact_heat_raw` | (4096,) f32 | 2D 热力直接采样（未修正） |
| `heatmap_2d` | (480,640) f32 | anchor 图像平面椭圆高斯热力图 |
| `anchor_object_mask` | (480,640) u8 | anchor 帧物体掩码 |
| `contact_frames` | (T,) i32 | 实际使用的接触帧 |
| 其他 | | `per_frame_point_evidence`、`aggregated_point_evidence`、`seed_mask` 等中间量 |

meta：`contact_heatmap_meta.json`（anchor_frame、seed_count、heat_area_ratio、椭圆参数、`frame_stats[].transform_source`）。
点云与 `contact_heat` **逐点对应**；`points_object_m` 与抓取标签的 `*_object` 量同一坐标系。

### 3.3 DINOv2 点特征 `dinov2_features/`

```text
point_dinov2_features.npy   (512, 384) f32   # 对应 sample_points 的 512 个采样点
point_pixels_uv.npy         (512, 2) i32     # 每个特征点的 anchor 帧像素
anchor_dinov2_grid.npz      features (35, 46, 384) f32  # anchor 图 dense patch grid
```

**注意**：点特征是 (512,384)，与 contact_heat 的 4096 点云**不是逐点对应**。
训练若要把特征附着到 4096 点云，需经 `pixels_uv` / `point_pixels_uv` 做像素最近邻映射，
或用 `anchor_dinov2_grid.npz` 按 `pixels_uv` 重新采样（双线性）。

### 3.4 抓取伪标签 `hamer_grasp_pseudo_label/grasp_pseudo_label.npz`

| 字段 | shape | 含义 |
|---|---|---|
| `T_grasp_cam` | (4,4) f32 | 夹爪 TCP 位姿（相机系）。旋转列：X=approach, Y=closing, Z=binormal；平移=TCP 位置 |
| `T_grasp_object` | (4,4) f32 | 同旋转；平移 = `tcp - object_center_camera`（与 `points_object_m` 同系） |
| `rotation_6d` | (6,) f32 | `T_grasp_object` 旋转的前两列（approach, closing）列优先展开 |
| `translation_object` | (3,) f32 | TCP 在物体系位置 |
| `width_m` | 标量 f32 | 夹爪开口 = ‖q_index − q_thumb‖ + 0.012 m |
| `q_thumb_object` / `q_index_object` | (3,) f32 | 两个指尖接触点（物体系） |
| `q_thumb_cam` / `q_index_cam` | (3,) f32 | 同上（相机系） |
| `candidate_T_cam` / `candidate_T_object` | (N,4,4) | 窗口内各帧候选 |
| `candidate_width_m` / `candidate_frames` | (N,) | 候选宽度 / 帧号 |
| `valid` | bool | quality != reject |
| `confidence` | 标量 f32 | 综合置信度 [0,1] |
| `selected_frame` | i32 | 代表帧；reject 时为 -1 |

**reject 约定**：所有位姿字段为 NaN，candidate 数组为空，`valid=False`，`confidence=0`，`selected_frame=-1`。
训练筛选用 `valid` 或 meta 的 `quality`（good/review/reject）过滤。

TCP 与接触点关系：`tcp = center - approach * 0.045`（`tcp_to_contact_offset`，yaml 可调），
即接触中心在夹爪内 approach 方向 4.5 cm 处。

### 3.5 抓取辅助文件

```text
hamer_grasp_pseudo_label/
├── grasp_pseudo_label_meta.json        # quality、confidence、per_frame 状态、consistency
├── selected_grasp_graspnet_row.npy     # (17,) GraspNet row：[score, width, 0.02, depth, R(9), t(3)]（相机系）
├── hamer_output/hamer_run_meta.json    # frames_requested/attempted/with_keypoints
├── hamer_output/skeleton2d/            # ViTPose 原图系 21 点 2D 手部关键点 *_hand_2d.npy (21,3)
└── viz/                                # selected_grasp_overlay_2d / window_candidates_2d / selected_grasp_3d
```

`skeleton2d/*_hand_2d.npy` 是原图像素坐标（x, y, conf），可直接配 D455 深度反投影；
`hamer_keypoints/*_pred_keypoints_2d.npy`（若存在）是 crop 归一化坐标，**不能**当原图像素用。

## 4. 用 Open3D 逐 episode 检查抓取位姿

推荐工具：

```bash
im2flow2act/bin/python tools/visualize_hamer_thumb_index_grasp_open3d.py \
  --episode-dir /media/ljian/lj/data_3d/hand_pouring_lfv/episode_0 \
  --show-candidates
```

窗口里显示：

- 灰色/洋红热力点云（magma）
- 紫色球 = thumb 接触点，绿色球 = index 接触点
- 红色线框 = 夹爪双指，蓝色线 = approach 方向
- TCP 处 RGB 坐标系
- 加 `--show-candidates` 会叠上窗口内所有候选（灰色）
- 加 `--object-frame` 切到物体系，方便跨 episode 比较

终端会打印关键诊断：

```text
quality: good,  valid: True,  confidence: 0.528
selected_frame: 51,  width_m: 0.0249
TCP          : [...]
approach     : [...]
depth_into_gripper (should be ~0.045): 0.0450
heat@nearest_point thumb=1.000 index=0.598
```

判断标准：

- `depth_into_gripper` 应 ≈ 0.045（等于 yaml 的 `tcp_to_contact_offset`）
- 紫色/绿色球应落在高热点上
- 红色夹爪线框应罩住杯把/把手等功能接触部位
- approach 方向不应扎进物体内部

批量抽查可用 shell 循环（每关一个窗口自动进下一个）：

```bash
for ep in /media/ljian/lj/data_3d/hand_pouring_lfv/episode_*; do
  echo "=== $(basename $ep) ==="
  im2flow2act/bin/python tools/visualize_hamer_thumb_index_grasp_open3d.py --episode-dir "$ep"
done
```

## 5. 已知注意事项

- HaMeR demo 依赖 Detectron2 人体检测；只露手臂的帧可能无输出 → 该 episode 进 review/reject。
  后续计划加"外部 hand bbox 入口"绕开人体检测（见 handoff 文档 Step 3）。
- approach 当前为 `-surface_normal`（近似 top-down 投影），尚未做多候选评分；改动会同步影响所有任务，需重跑 grasp 链。
- episode_61 类失败（contact_timing 缺 contact_start）是上游接触检测问题，先查 `contact_timing/contact_timing.json`。
- 每步都有 `<stage>_failed_logs.txt` 在 processed_root 下，批处理后先看它再看 check 汇总。
