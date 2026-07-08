# LFV 当前数据集训练与推理流程

本文档只说明当前 processed dataset 的训练和 validation 推理流程。仿真环境 scene 推理暂时不作为主流程，后续更换仿真环境后再单独规范。

## 1. 当前目标

当前要验证的是：

1. `/media/ljian/lj/data_3d/pickNplace_lfv` 这类已经处理好的 episode 数据能否正常被 dataset 读取。
2. 第一阶段 `GoalPoseDiffuser` 能否训练和从 validation split 抽样推理。
3. 第二阶段 `Full64 trajectory diffusion` 能否训练和从 validation split 抽样推理。

当前训练不需要准备仿真 scene，也不需要跑 `infer_goal_full64_saved_scenes.py`。

## 2. 代码入口

模型训练配置在：

```bash
configs/model/
```

当前需要关注的配置文件只有四个：

```bash
configs/model/task/goal_pose_multitask.yaml
configs/model/task/multitask_goal_full64.yaml
configs/model/train_goal_pose_diffusion.yaml
configs/model/train_dp3_goal_full64.yaml
```

dataset 统一在：

```bash
diffusion_policy_3d/dataset/lfv_dataset.py
```

训练脚本：

```bash
scripts/model/train_goal_pose_diffuser.py
scripts/model/train_dp3.py
```

validation dataset 推理脚本：

```bash
scripts/model/goal_pose_inference.py
scripts/model/infer_full64_dataset.py
```

## 3. 训练前必须根据实际情况修改的内容

### 3.1 数据集路径

第一阶段和第二阶段都要改：

```bash
configs/model/task/goal_pose_multitask.yaml
configs/model/task/multitask_goal_full64.yaml
```

字段：

```yaml
dataset:
  data_dirs:
    - "/media/ljian/lj/data_3d/pickNplace_lfv"
```
```yaml
task_name: pickNplace_goal_pose
dataset_name: pickNplace_lfv
output_name: pickNplace_lfv_goal_pose
```

这里必须指向已经完成数据处理的目录。每个 episode 至少需要包含：

```bash
episode_x/
  depth/
  meta.json
  sample_points/sampled_2d_uniform.npy
  target_sample_points/target_sampled_2d_uniform.npy
  se3_trajectory/dp_action_trajectory.npz
```

含义：

- `sample_points/sampled_2d_uniform.npy`：被操作物第一帧 2D 采样点。
- `target_sample_points/target_sampled_2d_uniform.npy`：目标物第一帧 2D 采样点。
- `depth/` 和 `meta.json`：反投影 3D 点云。
- `se3_trajectory/dp_action_trajectory.npz`：被操作物的 SE(3) 轨迹。

### 3.2 任务名字和 checkpoint 目录名

仍然在两个 task config 里改：

```yaml
task_name: pickNplace_goal_pose
dataset_name: pickNplace_lfv
output_name: pickNplace_lfv_goal_pose
instruction: "Place the cup on the plate"
```

第二阶段建议对应写成：

```yaml
task_name: pickNplace_full64
dataset_name: pickNplace_lfv
output_name: pickNplace_lfv_full64
instruction: "Place the cup on the plate"
```

字段含义：

- `task_name`：任务语义名，主要用于日志。
- `dataset_name`：数据集名字，主要用于人工阅读。
- `output_name`：checkpoint 目录名的核心字段，最重要。
- `instruction`：语言文本，用来生成 `lang_emb.npy`。

checkpoint 实际保存到：

```bash
<training.output_root>/<task.output_name>_seed<training.seed>/<timestamp>/checkpoints/
```

例如：

```bash
data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/20260708_153000/checkpoints/latest.ckpt
data/outputs/full64/pickNplace_lfv_full64_seed42/20260708_180000/checkpoints/latest.ckpt
```

训练脚本还会维护一个稳定软链接：

```bash
data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest
data/outputs/full64/pickNplace_lfv_full64_seed42/latest
```

所以推理时可以直接使用：

```bash
data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest/checkpoints/latest.ckpt
data/outputs/full64/pickNplace_lfv_full64_seed42/latest/checkpoints/latest.ckpt
```

### 3.3 相机内参来源

两个 task config 中都有：

```yaml
dataset:
  intrinsics_source: "depth_intrinsics_original"
```

当前代码会从每个 episode 的 `meta.json` 读取：

- `depth_scale`
- `depth_intrinsics_original`

如果后续确认你的 depth 已经严格对齐到 color，并且应该使用 color 相机内参，可以改成：

```yaml
intrinsics_source: "color_intrinsics"
```

当前默认先保持 `depth_intrinsics_original`。

### 3.4 语言文本和缺失语言 embedding

两个 task config 中都有：

```yaml
instruction: "Place the cup on the plate"

dataset:
  use_lang_emb: true
  missing_lang_emb: zero
```

如果 `lang_emb.npy` 不存在：

- `missing_lang_emb: zero`：用 `[1,1024]` 零向量，不中断训练。
- `missing_lang_emb: error`：缺失就报错，适合正式严格训练。
- `missing_lang_emb: none`：不补语言，仅在 `policy.use_lang_emb=false` 时使用。

建议当前先用：

```yaml
missing_lang_emb: zero
```

这样可以先验证训练流程。

### 3.5 checkpoint 根目录

第一阶段主配置：

```bash
configs/model/train_goal_pose_diffusion.yaml
```

字段：

```yaml
training:
  output_root: data/outputs/goal_pose
  seed: 42
  num_epochs: 1500
  checkpoint_every: 100
```

第二阶段主配置：

```bash
configs/model/train_dp3_goal_full64.yaml
```

字段：

```yaml
training:
  output_root: data/outputs/full64
  seed: 42
  num_epochs: 1500
  checkpoint_every: 100
```

如果要把 checkpoint 存到别的磁盘，可以改配置，也可以命令行覆盖：

```bash
training.output_root=/media/ljian/lj/checkpoints/goal_pose
training.output_root=/media/ljian/lj/checkpoints/full64
```

### 3.6 batch size 和 num_workers

第一阶段和第二阶段主配置都有：

```yaml
dataloader:
  batch_size: 64
  num_workers: 8

val_dataloader:
  batch_size: 64
  num_workers: 8
```

如果显存不够，先减小：

```bash
dataloader.batch_size=16 val_dataloader.batch_size=16
```

如果 DataLoader 或 zarr 读取有问题，先用：

```bash
dataloader.num_workers=0 val_dataloader.num_workers=0
```

## 4. Dataset 是如何被调用的

两个 task config 里通过 `_target_` 指定 dataset 类。

第一阶段：

```yaml
dataset:
  _target_: diffusion_policy_3d.dataset.lfv_dataset.GoalPoseSE3Dataset
```

第二阶段：

```yaml
dataset:
  _target_: diffusion_policy_3d.dataset.lfv_dataset.FullTrajectoryGoalConditionedSE3Dataset
```

训练脚本内部会执行：

```python
dataset = hydra.utils.instantiate(cfg.task.dataset)
val_dataset = dataset.get_validation_dataset()
```

因此训练读哪个数据集，完全由：

```yaml
configs/model/task/*.yaml -> dataset.data_dirs
```

控制。

train/val 切分由：

```yaml
dataset:
  val_ratio: 0.1
```

控制。当前是固定随机种子切分 episode。

## 5. 执行训练前的检查

先确认数据处理结果存在：

```bash
ls /media/ljian/lj/data_3d/pickNplace_lfv/episode_0
```

至少应该看到：

```bash
depth
meta.json
sample_points
target_sample_points
se3_trajectory
```

如果还没跑完整数据处理：

```bash
cd /home/users1/ljian/LFV
python scripts/run_pipeline.py \
  --config configs/pipeline/picknplace.yaml \
  --steps prepare,dino,sam2,sample,track,se3
```

生成语言 embedding：

```bash
cd /home/users1/ljian/LFV
python -m diffusion_policy_3d.dataset.generate_lang_emb \
  --task-config configs/model/task/goal_pose_multitask.yaml
```

该命令会读取 task config 中的：

```yaml
instruction
dataset.data_dirs[0]
```

并保存：

```bash
/media/ljian/lj/data_3d/pickNplace_lfv/lang_emb.npy
```

## 6. Smoke Test

正式训练前，建议先各跑一次 1 step，确认 dataset、model、checkpoint 都通。

第一阶段 smoke test：

```bash
cd /home/users1/ljian/LFV
python scripts/model/train_goal_pose_diffuser.py \
  --config-name train_goal_pose_diffusion \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0
```

第二阶段 smoke test：

```bash
cd /home/users1/ljian/LFV
python scripts/model/train_dp3.py \
  --config-name train_dp3_goal_full64 \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0
```

如果 smoke test 都能保存 `latest.ckpt`，再进入正式训练。

## 7. 正式训练

第一阶段训练：

```bash
cd /home/users1/ljian/LFV
python scripts/model/train_goal_pose_diffuser.py \
  --config-name train_goal_pose_diffusion
```

第一阶段默认 checkpoint：

```bash
data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest/checkpoints/latest.ckpt
```

第二阶段训练：

```bash
cd /home/users1/ljian/LFV
python scripts/model/train_dp3.py \
  --config-name train_dp3_goal_full64
```

第二阶段默认 checkpoint：

```bash
data/outputs/full64/pickNplace_lfv_full64_seed42/latest/checkpoints/latest.ckpt
```

如果要指定输出根目录：

```bash
python scripts/model/train_goal_pose_diffuser.py \
  --config-name train_goal_pose_diffusion \
  training.output_root=/media/ljian/lj/checkpoints/goal_pose

python scripts/model/train_dp3.py \
  --config-name train_dp3_goal_full64 \
  training.output_root=/media/ljian/lj/checkpoints/full64
```

如果中断后续训：

```bash
python scripts/model/train_goal_pose_diffuser.py \
  --config-name train_goal_pose_diffusion \
  training.resume=true

python scripts/model/train_dp3.py \
  --config-name train_dp3_goal_full64 \
  training.resume=true
```

`resume=true` 会在对应 `<output_root>/<output_name>_seed<seed>/` 下寻找最新 timestamp 目录。

## 8. 从当前数据集做推理验证

当前推理验证只从 validation dataset 抽样，不依赖仿真环境。

### 8.1 第一阶段 goal pose 推理

默认读取：

```bash
data/outputs/goal_pose/pickNplace_lfv_goal_pose_seed42/latest/checkpoints/latest.ckpt
```

执行：

```bash
cd /home/users1/ljian/LFV
python scripts/model/goal_pose_inference.py
```

指定 checkpoint：

```bash
LFV_GOAL_CKPT=/path/to/goal_pose.ckpt \
python scripts/model/goal_pose_inference.py
```

该脚本默认从 validation dataset 取 5 个样本，保存 overlay 和 `summary.csv`。

### 8.2 第二阶段 full64 轨迹推理

默认读取：

```bash
data/outputs/full64/pickNplace_lfv_full64_seed42/latest/checkpoints/latest.ckpt
```

执行：

```bash
cd /home/users1/ljian/LFV
python scripts/model/infer_full64_dataset.py
```

指定 checkpoint、样本数和输出目录：

```bash
LFV_FULL64_CKPT=data/outputs/full64/pickNplace_lfv_full64_seed42/latest/checkpoints/latest.ckpt \
LFV_NUM_DATASET_SAMPLES=5 \
LFV_FULL64_DATASET_OUTPUT_DIR=data/outputs/full64/debug_val_samples \
python scripts/model/infer_full64_dataset.py
```

输出内容：

```bash
sample_000.npz
sample_000_pred_overlay.png
sample_000_gt_overlay.png
sample_000_traj.png
sample_001.npz
...
summary.csv
```

每个 `.npz` 包含：

- `pred_action_full`
- `pred_action_exec`
- `gt_action`
- `centroid_0`
- `traj_idx`
- `mean_trans_err_cm`
- `max_trans_err_cm`

`sample_000_pred_overlay.png` 和 `sample_000_gt_overlay.png` 是第一帧 RGB 图像上的轨迹 overlay：

- 每个轨迹点画成一个 3D 坐标系。
- 坐标系原点之间用黄色线连接。
- Pred 和 GT 分开保存，避免 64 个坐标系叠在一张图上过于混乱。

`sample_000_traj.png` 是辅助 3D 曲线图：

- 左侧：GT 和 Pred 的 3D xyz 轨迹。
- 右侧：x/y/z 随 step 的曲线，以及每一步 translation error。

如果只想保存数值、不保存图片：

```bash
LFV_SAVE_TRAJ_PLOTS=0 python scripts/model/infer_full64_dataset.py
```

如果只想关闭第一帧图像 overlay：

```bash
LFV_SAVE_IMAGE_OVERLAYS=0 python scripts/model/infer_full64_dataset.py
```

## 9. 推荐执行顺序

1. 修改两个 task config：
   - `dataset.data_dirs`
   - `task_name`
   - `dataset_name`
   - `output_name`
   - `instruction`
   - `intrinsics_source`

2. 必要时修改两个 train config：
   - `training.output_root`
   - `training.seed`
   - `training.num_epochs`
   - `training.checkpoint_every`
   - `dataloader.batch_size`
   - `dataloader.num_workers`

3. 确认数据处理产物存在。

4. 生成 `lang_emb.npy`。

5. 跑第一阶段 smoke test。

6. 跑第二阶段 smoke test。

7. 正式训练第一阶段。

8. 用 validation dataset 推理第一阶段。

9. 正式训练第二阶段。

10. 用 validation dataset 推理第二阶段。

## 10. 当前不作为主流程的内容

以下脚本暂时不是当前训练验证主流程：

```bash
scripts/model/infer_goal_pose_saved_scenes.py
scripts/model/infer_goal_full64_saved_scenes.py
```

它们是给后续 saved scene / 仿真环境输入使用的。因为你后续会换仿真环境，所以当前先不要依赖它们判断模型训练是否正常。
