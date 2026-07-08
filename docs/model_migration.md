# LFV Model Migration Notes

本文档记录从 `/home/users1/ljian/object_centric_diffusion` 迁移到
`/home/users1/ljian/LFV` 的模型、dataset、训练和推理代码范围，以及这些代码和 LFV
数据处理管道的连接方式。
## 迁移目标

LFV 工程现在包含两部分：

1. 数据处理管道：DINO 检测、SAM2 分割、2D 采样、RGB-D 反投影、TAPIP3D 点云跟踪、SE(3) 轨迹生成。
2. 后续模型代码：dataset、模型结构、训练入口、可视化和推理入口。

模型训练使用的数据根目录默认是：

```bash
/media/ljian/lj/data_3d/pickNplace_lfv
```

该目录可以是 `/media/ljian/lj/new_data/pickNplace` 的软链接，也可以是数据处理管道输出后的真实目录。训练代码只关心处理后的 episode 文件结构。

## 已迁移文件

### 核心 Python package

源路径：

```bash
/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d
```

目标路径：

```bash
/home/users1/ljian/LFV/diffusion_policy_3d
```

迁移内容和作用：

- `diffusion_policy_3d/dataset`
  - `lfv_dataset.py`：统一后的 LFV dataset 文件，包含公共相机内参读取、深度反投影、SE(3) 工具函数、第一阶段 `GoalPoseSE3Dataset` 和第二阶段 `FullTrajectoryGoalConditionedSE3Dataset`。
  - `generate_lang_emb.py`：用 CLIP 为任务文本生成 `lang_emb.npy`。
  - 未被当前两阶段训练配置使用的 RLBench dataset、旧单任务 dataset 和拆分前的 dataset 文件没有保留。

- `diffusion_policy_3d/policy`
  - `goal_pose_diffuser.py`：第一阶段 goal pose diffusion policy。
  - `simple_dp3.py`、`dp3.py`：第二阶段轨迹扩散 policy 和 DP3 相关 policy。
  - `base_policy.py`：policy 基类。

- `diffusion_policy_3d/model`
  - `diffusion/*`：1D diffusion UNet、EMA、mask generator、scheduler 相关组件。
  - `vision/*`：点云编码、cross attention / set transformer 相关编码器。
  - `goal/*`：goal pose 编码和 pose 表示转换。
  - `clip/*`：语言 embedding 所需的 CLIP 代码。
  - `common/*`：normalizer、tensor 工具、学习率调度等公共组件。

- `diffusion_policy_3d/common`
  - checkpoint、logger、sampler、replay buffer 等训练通用工具。

### 训练和推理脚本

源路径：

```bash
/home/users1/ljian/object_centric_diffusion/tools
```

目标路径：

```bash
/home/users1/ljian/LFV/scripts/model
```

迁移内容和作用：

- `train_goal_pose_diffuser.py`
  - 第一阶段训练入口。
  - 使用 `configs/model/train_goal_pose_diffusion.yaml` 和 `configs/model/task/goal_pose_multitask.yaml`。

- `train_dp3.py`
  - 第二阶段 full64 trajectory diffusion 训练入口。
  - 使用 `configs/model/train_dp3_goal_full64.yaml` 和 `configs/model/task/multitask_goal_full64.yaml`。

- `vis_goal_pose_diffuser.py`
  - 第一阶段 checkpoint 可视化入口。
  - 运行前需要把脚本顶部的 `CKPT_PATH` 改成实际训练输出。

- `goal_pose_inference.py`
  - 第一阶段 goal pose 推理入口。
  - 运行前需要把脚本顶部的 `CKPT_PATH` 改成实际训练输出。

- `infer_goal_full64_saved_scenes.py`
  - 第二阶段 saved scene full64 轨迹推理入口。
  - 已改为从每个 scene/episode 的 `meta.json` 读取相机内参和 `depth_scale`。
  - 运行前需要准备 `SCENE_ROOT` 下的 scene 数据和第一阶段输出。

这些脚本的 `ROOT_DIR` 已改为 `/home/users1/ljian/LFV`，Hydra 的 `config_path`
也指向 LFV 内部的 `configs/model` 目录。

### 配置文件

源路径：

```bash
/home/users1/ljian/object_centric_diffusion/config
```

目标路径：

```bash
/home/users1/ljian/LFV/configs/model
```

迁移内容和作用：

- `configs/model/train_goal_pose_diffusion.yaml`
  - 第一阶段 GoalPoseDiffuser 训练主配置。

- `configs/model/train_dp3_goal_full64.yaml`
  - 第二阶段 full64 trajectory diffusion 训练主配置。

- `configs/model/task/goal_pose_multitask.yaml`
  - 第一阶段 dataset 和 shape 配置。
  - Hydra target 是 `diffusion_policy_3d.dataset.lfv_dataset.GoalPoseSE3Dataset`。
  - 当前默认 `data_dirs` 是 `/media/ljian/lj/data_3d/pickNplace_lfv`。
  - 当前默认 `intrinsics_source` 是 `depth_intrinsics_original`。

- `configs/model/task/multitask_goal_full64.yaml`
  - 第二阶段 full64 dataset 和 shape 配置。
  - Hydra target 是 `diffusion_policy_3d.dataset.lfv_dataset.FullTrajectoryGoalConditionedSE3Dataset`。
  - 当前默认 `data_dirs` 是 `/media/ljian/lj/data_3d/pickNplace_lfv`。
  - 当前默认 `intrinsics_source` 是 `depth_intrinsics_original`。

### 工具函数

源路径：

```bash
/home/users1/ljian/object_centric_diffusion/utils
```

目标路径：

```bash
/home/users1/ljian/LFV/utils
```

迁移内容：

- `pose_utils.py`
- `transform_utils.py`
- `io_utils.py`
- `vis_utils.py`

这些文件用于位姿变换、I/O 和可视化辅助。迁移时只复制模型和脚本会引用到的通用工具。

## 没有迁移的内容

以下内容没有迁移到 LFV：

- `env`、`env_real`、`env_rlbench_peract` 等仿真/真实机器人环境代码。
- MuJoCo、RLBench、FoundationPose 的完整资产和环境数据。
- 历史训练输出、wandb 输出、checkpoint 输出目录。
- `object_centric_diffusion/env_data` 下的旧 saved scene 数据。

原因是当前 LFV 的目标是把真实 RGB-D 数据处理结果接到 goal pose / trajectory diffusion
训练和推理，不需要完整仿真环境资产。checkpoint 和大体积数据也不应该作为工程源码迁移。

## 已做的适配

### 相机内参

旧代码中部分 dataset 使用：

```bash
/home/users1/ljian/im2Flow2Act/data_local/simulation/instrinsic_5-1.pkl
```

这已经不适用于 D455 新采集数据。现在训练 dataset 改为读取每个 episode 的：

```bash
episode_x/meta.json
```

默认字段：

```yaml
intrinsics_source: depth_intrinsics_original
```

读取逻辑：

- 从 `meta.json` 读取 `depth_intrinsics_original`。
- 从 `meta.json` 读取 `depth_scale`。
- 读取 `depth/0` 后执行 `depth_meter = depth_uint16 * depth_scale`。
- 用该深度和 episode 内参把 sampled 2D points 反投影成点云。

如果后续确认应该用对齐到 color 平面的 color 内参，可以在两个 task config 里把：

```yaml
intrinsics_source: "depth_intrinsics_original"
```

改成：

```yaml
intrinsics_source: "color_intrinsics"
```

### 数据文件结构

训练前，每个 episode 至少需要包含：

```bash
episode_x/
  depth/
  meta.json
  sample_points/sampled_2d_uniform.npy
  target_sample_points/target_sampled_2d_uniform.npy
  se3_trajectory/dp_action_trajectory.npz
```

如果 `use_lang_emb: true`，数据根目录最好包含：

```bash
lang_emb.npy
```

可以用：

```bash
python -m diffusion_policy_3d.dataset.generate_lang_emb
```

在 `/media/ljian/lj/data_3d/pickNplace_lfv/lang_emb.npy` 生成默认文本 embedding。需要修改任务语言时，改
`diffusion_policy_3d/dataset/generate_lang_emb.py` 里的文本字符串。

## 推荐运行顺序

### 1. 完成 LFV 数据处理

在 LFV 根目录运行数据处理管道，确保每个 episode 有 bbox、mask、sample points、track 和 SE(3) trajectory。

```bash
cd /home/users1/ljian/LFV
python scripts/run_pipeline.py --config configs/pipeline/picknplace.yaml --steps prepare,dino,sam2,sample,track,se3
```

如果某一步已经完成，可以只跑缺失步骤，例如：

```bash
python scripts/run_pipeline.py --config configs/pipeline/picknplace.yaml --steps se3
```

### 2. 生成语言 embedding

```bash
cd /home/users1/ljian/LFV
python -m diffusion_policy_3d.dataset.generate_lang_emb
```

### 3. 训练第一阶段 GoalPoseDiffuser

```bash
cd /home/users1/ljian/LFV
python scripts/model/train_goal_pose_diffuser.py --config-name train_goal_pose_diffusion
```

输出默认在：

```bash
data/outputs_local_goal_pose
```

### 4. 第一阶段可视化或推理

先修改脚本中的 checkpoint：

```bash
scripts/model/vis_goal_pose_diffuser.py
scripts/model/goal_pose_inference.py
```

然后运行：

```bash
python scripts/model/vis_goal_pose_diffuser.py
python scripts/model/goal_pose_inference.py
```

### 5. 训练第二阶段 full64 trajectory diffusion

```bash
cd /home/users1/ljian/LFV
python scripts/model/train_dp3.py --config-name train_dp3_goal_full64
```

第二阶段读取同一批处理后的 episode，并使用第一帧 manipulated/target 点云、完整 SE(3) 轨迹和 goal-conditioned full64 标签。

### 6. 第二阶段 saved scene 推理

运行前需要准备 `SCENE_ROOT`，并确保每个 scene/episode 有：

- `meta.json`
- RGB 图像
- depth
- affordance/target sample points
- 第一阶段输出的 `pred_goal_pose7d.npy`

然后修改：

```bash
scripts/model/infer_goal_full64_saved_scenes.py
```

中的：

```python
SECOND_STAGE_CKPT_PATH = "..."
SCENE_ROOT = "..."
```

再运行：

```bash
python scripts/model/infer_goal_full64_saved_scenes.py
```

## 当前迁移方式

迁移使用文件级复制，排除了 `__pycache__` 和 `.pyc`，避免把运行缓存带入新工程。迁移后做了以下代码适配：

- 训练脚本根目录从旧工程改为 LFV 根目录。
- Hydra config 路径改为 LFV 内部 `config`。
- 第一阶段和第二阶段 dataset 的数据目录改为 `pickNplace_lfv`。
- 训练 dataset 的反投影内参改为从每个 episode 的 `meta.json` 读取。
- 第二阶段 saved scene 推理的反投影内参也改为从每个 scene/episode 的 `meta.json` 读取。

## 注意事项

- 没有迁移 checkpoint。训练完成后，推理脚本需要手动填写 LFV 下新 checkpoint 的路径。
- 没有迁移旧 `env_data`。如果要跑 saved scene 推理，需要把对应 scene 数据准备到 LFV 约定目录，或者修改 `SCENE_ROOT`。
- 如果运行时报 `ModuleNotFoundError`，说明当前 conda 环境缺少模型训练依赖，需要在训练环境里安装对应包，例如 `scipy`、`hydra-core`、`diffusers`、`zarr`、`dill` 等。
