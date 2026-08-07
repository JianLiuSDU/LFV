# A3b 跨数据集受控训练与仿真推理记录

## 1. 实验目的与边界

本轮实验固定 A3b 网络结构、损失、优化器和采样器，只改变训练数据，回答“轨迹执行失败主要来自网络，还是来自数据标签与仿真输入分布”这一问题。实验包含两条互不混用的数据链：

1. 旧 pouring：`/media/ljian/lj/data_3d/pouring`；
2. pick-and-place：`/media/ljian/lj/data_3d/pickNplace_lfv`。

本轮没有修改 `ThreeTokenHierarchicalDiffusion`、共享 scene encoder、Goal Diffusion 或 Trajectory Diffusion 的计算结构。代码变化只包括旧数据格式兼容、数据缓存配置、通用仿真任务适配和结果可视化标签。

## 2. 固定的 A3b 网络

模型注册名为 `three_token_hierarchical_diffusion`，两次训练使用完全相同的参数：

- `hidden_dim=128`，4 个 attention heads；
- 共享 three-token scene encoder；
- Goal Diffusion：4 层 Transformer，状态为 9D 终态位姿；
- Trajectory Diffusion：6 层 Transformer，状态为 64×9D 累积位姿轨迹；
- 100 个 DDPM 训练时间步，20 步推理；
- hard start token；
- 离散 sinusoidal 时间位置编码；
- 2 层 gated goal-context interaction；
- 4 个 phase tokens，`sigma=0.22`，phase residual gate 开启；
- full temporal attention；
- AdamW、AMP、EMA、梯度裁剪、cosine learning-rate schedule；
- Goal 与 Trajectory 同时训练，共享 encoder 接收两部分损失的梯度。

模型输入和输出保持：

```text
manipulated_points [B,256,3]       manipulated_dino [B,256,384]
reference_points   [B,256,3]       reference_dino   [B,256,384]
                         │
                         ▼
              shared three-token encoder
                 │                    │
                 ▼                    ▼
       Goal Diffusion [B,K,9]   Trajectory Diffusion
                                  [B,K,M,64,9]
```

9D 位姿为 3D translation + continuous 6D rotation；64 帧轨迹中的每一帧都是相对第 0 帧的累积变换 `T(0→k)`，不是相邻帧残差。

## 3. 数据改造

### 3.1 旧数据格式兼容

旧 pouring episode 不包含新格式的 `meta.json`。新增 `lfv/datasets/functional_motion/source_io.py`，按以下顺序读取标定：

1. 新格式优先使用 `meta.json`；
2. 旧格式回退到 `point_tracking/tapip3d_result.npz` 中的 intrinsic；
3. 旧 Zarr depth 已经是米，不再次应用深度 scale；
4. audit 与 cache builder 共用同一个标定入口，防止“审计通过但缓存读取方式不同”。

该修改是 source adapter，不改变 A3b 模型。

### 3.2 统一重采样与 DINO

两套数据都重新生成独立缓存：

- manipulated mask 内采样 256 个有效深度像素；
- reference mask 内采样 256 个有效深度像素；
- 同一像素索引同时用于 RGB 上的 DINOv2 特征采样和 depth 反投影；
- 每组 256 个索引唯一；
- DINOv2 `ViT-S/14` 离线特征维度为 384；
- 保存 `goal_pose9d [9]` 和 `trajectory_pose9d [64,9]`。

缓存不会写回原始数据目录：

| 数据 | cache | 有效 episode | train/val/test |
|---|---|---:|---:|
| old pouring | `/home/users1/ljian/lfv_data_cache/stage2/pouring_old_a3b_v1` | 82/82 | 66/8/8 |
| pickNplace_lfv | `/home/users1/ljian/lfv_data_cache/stage2/picknplace_lfv_a3b_v1` | 180/180 | 144/18/18 |

原数据没有可靠的跨 episode `object_instance_id`，因此文档明确把当前拆分标为 `episode_split_baseline`；拆分使用固定 seed=42 并保存到 `split_manifest.json`，训练和测试不混用 episode。

## 4. 数据标签诊断

对训练集中每条数据计算：

```text
terminal_to_reference = goal_translation - mean(reference_points)
```

该量表示预测终态的 manipulated 原点与 reference 点云中心之间的剩余向量。结果为：

| 数据 | residual mean xyz (m) | residual std xyz (m) | residual norm median / p90 (m) |
|---|---|---|---|
| old pouring | `[-0.0753,-0.1464,-0.1306]` | `[0.0116,0.0195,0.0163]` | `0.2147 / 0.2344` |
| pouring_lfv（此前 A3b） | `[-0.0659,-0.0615,-0.0770]` | `[0.0142,0.0220,0.0214]` | `0.1209 / 0.1456` |
| pickNplace_lfv | `[-0.0083,-0.0118,-0.0271]` | `[0.0082,0.0072,0.0082]` | `0.0323 / 0.0410` |

旧 pouring 的终态标签平均仍离 reference bowl 中心约 21.5 cm。训练样本 GT 投影也显示轨迹从杯子向画面中央/小黄鸭附近移动，而不是收敛到红色 bowl。网络可以很好地拟合该轨迹，但这不等价于学会“把杯子移动到 bowl 上方”。该结果是数据语义问题的直接证据。

相比之下，pickNplace_lfv 的终态距离 plate 中心中位数约 3.2 cm，符合把香蕉放到盘子上的任务定义。

## 5. old pouring A3b 结果

训练配置：`configs/stage2/a3b_pouring_old.yaml`。

- 训练 242 轮后 early stopping；
- best checkpoint：epoch 161；
- best validation total loss：`0.441394`；
- checkpoint：`/home/users1/ljian/lfv_runs/stage2/controlled_dataset_comparison/a3b_pouring_old/checkpoints/best.pt`。

保留 test split、16 个 Goal、每个 Goal 2 条 Trajectory、EMA 权重：

| 指标 | 结果 |
|---|---:|
| Goal top-1 translation | 3.731 cm |
| Goal top-1 rotation | 13.544° |
| Goal best translation | 3.703 cm |
| Trajectory top-1 mean translation | 4.308 cm |
| Trajectory top-1 mean rotation | 10.552° |
| Trajectory top-1 endpoint translation | 3.779 cm |
| First-step translation error | 2.703 mm |

固定蓝杯仿真使用相同 RGB-D snapshot，预测终点为 `[-0.0750, 0.0124, 0.1889] m`，可见 bowl 点云中心为 `[0.0454, 0.1878, 0.0349] m`，二者平面距离 21.28 cm、3D 距离 26.26 cm。候选相对训练先验的 `goal_residual_z2=12.30`，说明仿真输入/目标关系也落在训练分布之外。

主要结果：

- test metrics：`.../a3b_pouring_old/test_metrics.json`；
- 训练样本 GT 对比：`.../a3b_pouring_old/train_visualization/train_inference_gt_vs_top1_summary.png`；
- 蓝杯仿真汇总：`.../a3b_pouring_old/sim_blue_mug_seed_0/simulation_inference_summary.png`。

## 6. pick-and-place A3b 结果

训练配置：`configs/stage2/a3b_picknplace_lfv.yaml`。

- 训练 227 轮后 early stopping；
- best checkpoint：epoch 146；
- best validation total loss：`0.248464`；
- checkpoint：`/home/users1/ljian/lfv_runs/stage2/controlled_dataset_comparison/a3b_picknplace_lfv/checkpoints/best.pt`。

保留 test split、16 个 Goal、每个 Goal 2 条 Trajectory、EMA 权重：

| 指标 | 结果 |
|---|---:|
| Goal top-1 translation | 2.912 cm |
| Goal top-1 rotation | 12.563° |
| Goal best translation | 2.888 cm |
| Trajectory top-1 mean translation | 3.384 cm |
| Trajectory top-1 mean rotation | 8.856° |
| Trajectory top-1 endpoint translation | 2.724 cm |
| First-step translation error | 1.398 mm |

训练样本可视化中，GT 和预测轨迹都从香蕉移动到 plate；四个固定样本的 Goal top-1 translation 分别为 0.42、1.78、3.41、0.64 cm。这与旧 pouring 的错误 reference 关系形成明确对照。

主要结果：

- test metrics：`.../a3b_picknplace_lfv/test_metrics.json`；
- 训练样本 GT 对比：`.../a3b_picknplace_lfv/train_visualization/train_inference_gt_vs_top1_summary.png`；
- 终态 GT 对比：`.../a3b_picknplace_lfv/train_visualization/train_goal_pose_gt_vs_top1_summary.png`；
- 仿真汇总：`.../a3b_picknplace_lfv/sim_banana_plate_seed_0/simulation_inference_summary.png`。

## 7. 香蕉—盘子仿真输入

新增 `LFVPickBananaPlate-v1`，只承担 task adapter：

- manipulated：YCB `011_banana`；
- reference：YCB `029_plate`；
- 香蕉位于 base-camera 图像左侧，盘子位于右侧；
- 二者中心间距为 0.40 m，香蕉近似纵向；
- YCB plate 缩放为 `0.72×`；对齐后仿真相机点云关系向量为 `[0.3976,-0.0003,0.0268] m`，训练中位数为 `[0.4162,0.0038,0.0140] m`；
- 对齐后 plate 点云 extent 为 `[0.1847,0.1223,0.1370] m`，训练中位数为 `[0.1855,0.1242,0.1460] m`；
- 输出 RGB、metric depth、intrinsic、instance masks、两组相机点云以及 actor/world/camera 变换；
- 推理仍走与 cache builder 相同的联合 pixel/XYZ/DINO 采样路径。

固定快照：`/home/users1/ljian/lfv_runs/stage2/controlled_dataset_comparison/picknplace_banana_plate_seed_0/snapshot/task_snapshot.npz`。

分布对齐前，仿真 `goal_residual_z2=7.49`，预测终态到 plate 可见中心的 3D 距离为 9.42 cm。对齐后同一 checkpoint、同一采样 seed 下分别改善为 `3.83` 和 7.60 cm；平面距离为 7.29 cm，位于缩放后 plate 的约 9.2 cm 半径覆盖范围内，终态高度比可见 plate 中心高约 2.18 cm。二维轨迹和终态坐标系均进入 plate 区域，但首步 2.65 mm 仍高于训练 p95 的 1.62 mm，因此当前只报告“合理推理”，不宣称已经通过机器人执行成功率验证。

## 8. 当前结论

1. A3b 在旧 pouring 上可以稳定收敛，说明代码和优化链路能工作；
2. 旧 pouring 标签中的终态—reference 关系与 pouring 任务语义不一致，换成新网络不会自动修复标签；
3. 因此此前“到不了 bowl 正上方”至少包含明确的数据因素，不能只归因于 trajectory Transformer；
4. pickNplace_lfv 在相同模型下得到正确的目标语义，且仿真经过输入几何对齐后明显改善，证明数据标签和输入分布都是主要因素；
5. pick&place 仍有 7.60 cm 的三维中心偏差和偏大的首步，说明不能把剩余误差全部归咎于数据，模型的跨域泛化仍需后续独立研究；
6. 本轮严格保留 A3b，不以看到仿真失败为理由临时改模型，确保对比结论有效。

## 9. 固定复现实验入口

```bash
# 训练
python scripts/stage2/train.py --config configs/stage2/a3b_pouring_old.yaml
python scripts/stage2/train.py --config configs/stage2/a3b_picknplace_lfv.yaml

# 保留测试集评估
python scripts/stage2/evaluate.py --checkpoint <best.pt> \
  --cache-root <cache_root> --split test --output <test_metrics.json> \
  --device cuda --num-goals 16 --num-trajectories 2

# 通用仿真快照与推理
python scripts/sim/export_task_snapshot.py --task picknplace \
  --output-dir <snapshot_dir> --seed 0
python scripts/stage2/infer_sim_snapshot.py --task picknplace \
  --manipulated-label banana --reference-label plate \
  --checkpoint <best.pt> --snapshot <task_snapshot.npz> \
  --output-dir <inference_dir> --device cuda \
  --num-goals 16 --num-trajectories 2
```

代码检查结果：

- `pytest -q tests/stage2`：27 passed；
- ManiSkill scene、snapshot exporter、Stage2 simulator inference 均通过 `py_compile`；
- checkpoint、缓存、DINO 权重和运行图像均保存在仓库外，没有提交大文件到 Git。
