# Motion Functional Field 梯度路由实验 A/B

本文档记录第二阶段在 `pouring_lfv` 上进行的第一版功能运动场消融。实验目的不是增加新的网络模块，而是回答一个直接问题：**运动场的 relevance head 是否应同时由 Goal 和 Trajectory 两个扩散分支监督，还是只由终态 Goal 分支监督？** 该实验保留当前三 token encoder、Goal Pose Diffusion 和 Goal-conditioned Trajectory Diffusion 的全部结构，只改变轨迹损失对运动场加权池化的梯度路径。

## 1. 实验定义

### 实验 A：Joint Field（主模型）

配置 `configs/stage2/motion_field_v2_pouring_lfv_joint_ablation.yaml`，`motion_field_gradient_mode: joint`。场权重参与 Goal 和 Trajectory 两个分支的前向计算，两个损失都可以反向更新 relevance head。因此，场学习到的是同时有利于终态位姿和中间运动生成的任务相关性。

### 实验 B：Goal-only Field（梯度消融）

配置 `configs/stage2/motion_field_v2_pouring_lfv_goal_only_field.yaml`，仅加入 `motion_field_gradient_mode: goal_only`。编码器产生的数值与 A 完全相同；在构造 trajectory context 时，仅对场加权后的 `H_m/H_r` 做 `detach`，所以：

* relevance head 只接收 Goal loss 的梯度；
* PointNet/关系特征本身仍可接收两个分支的梯度；
* Trajectory decoder、goal-context 和 temporal blocks 的训练方式不变；
* 推理阶段没有 detach，A/B 的采样接口完全一致。

这使 B 成为干净的“监督来源”消融，而不是改变模型容量或输入信息的对照组。单元测试 `test_goal_only_field_detaches_relevance_from_trajectory_loss` 验证了两种模式前向值一致且梯度路由符合定义。

## 2. 数据、划分与训练控制

两组实验均使用同一个缓存：`/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1`，对应原始数据 `/media/ljian/lj/data_3d/pouring_lfv`。按 episode/object-instance 的固定划分使用 18 个测试 episode；训练、验证和测试样本没有跨 split 混用。随机种子为 42，点顺序随机置换策略、DINO 维度 384、hidden dimension 128、4 heads、Goal 4 层、Trajectory 6 层、DDPM 训练步数 100、DDIM 推理 20 步及所有优化器/损失权重均保持一致。两组均使用 AdamW、EMA (`0.995`)、梯度裁剪 (`1.0`)、AMP 配置（本次机器无 CUDA 时自动以 CPU 运行）和 early stopping。

完整主模型 A 使用已有完整训练结果：

`/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/checkpoints/best.pt`（best epoch 129）。

B 从相同数据和随机种子独立训练，best checkpoint 为：

`/home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/checkpoints/best.pt`（best epoch 66，global step 603）。

`motion_field_ablation/joint` 下曾启动过重复的 A 训练，但为避免 CPU 上重复计算在 epoch 9 主动停止；该目录不作为 A 的结果。

训练曲线中的最佳验证总损失分别为 A `0.47085`（epoch 129）和 B `0.48698`（epoch 66）；二者的损失项定义和权重完全相同，因此 B 的轻微验证退化可归因于梯度路由而非额外正则或网络容量变化。

## 3. 测试指标（EMA，K=4）

下表来自固定 test split、相同采样协议（4 个 goal，每个 goal 1 条 trajectory）。平移单位为米，旋转为 SO(3) 测地角度。

| 指标 | A Joint | B Goal-only | B−A |
|---|---:|---:|---:|
| Goal top-1 平移误差 | 0.02925 | 0.03283 | +0.00358 |
| Goal top-1 旋转误差 (deg) | 25.72 | **23.68** | −2.04 |
| Trajectory top-1 平移误差 | **0.04288** | 0.04687 | +0.00399 |
| Trajectory top-1 旋转误差 (deg) | **14.04** | 14.35 | +0.31 |
| Trajectory endpoint 平移误差 | **0.03041** | 0.03525 | +0.00483 |
| Trajectory endpoint 旋转误差 (deg) | 25.60 | **23.63** | −1.97 |
| 第一帧平移误差 | **0.00267** | 0.00333 | +0.00066 |

完整 JSON：

* A：[test_metrics_cpu.json](file:///home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/test_metrics_cpu.json)
* B：[test_metrics.json](file:///home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/test_metrics.json)

结果表明 B 的终态旋转略有改善，但牺牲了终态平移、轨迹平移和端点误差；因此在当前数据规模和损失权重下，不能据此替换联合监督的 A。

## 4. 运动场是否真的被使用：干预实验

对测试输入分别使用 learned field、uniform field（所有点等权）和 rolled field（沿点序滚动场值）。如果场只是一种无关的缩放，替换后指标不应系统性变差；如果场承载任务相关性，破坏其空间分布应造成退化。

| 模式 | A Goal 平移 | A Traj 平移 | B Goal 平移 | B Traj 平移 |
|---|---:|---:|---:|---:|
| learned | 0.02932 | 0.04198 | 0.03362 | 0.04651 |
| uniform | 0.03302 | 0.04357 | 0.03755 | 0.04792 |
| rolled | 0.03305 | 0.04355 | 0.03780 | 0.04804 |

相对于 learned，A 的 uniform/rolled 分别使 Goal 平移误差增加约 3.70/3.73 mm、轨迹平移误差增加约 1.60/1.57 mm；B 分别增加约 3.93/4.18 mm 和 1.41/1.53 mm。两组实验都显示场具有因果作用，而不是仅用于可视化。B 的测试场统计为 manipulated entropy `0.9596`、peak mass `0.01247`；A 为 entropy `0.9293`、peak mass `0.01986`。B 的场在测试集上更接近均匀，说明仅用 Goal 监督会减弱对中间运动相关区域的选择性。

完整干预记录：

* A：[causality.json](file:///home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/causality.json)
* B：[causality.json](file:///home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/causality.json)

## 5. 频谱与平滑性结果

在 GT goal 条件下评估轨迹的频段能量保留率。A 的位置低/中/高频保留率为 `0.887/0.339/0.195`，速度低/中/高频为 `0.799/0.245/0.151`；B 为 `0.672/0.233/0.403` 和 `0.591/0.222/0.278`。B 的低频保留明显下降，说明 Goal-only 场监督没有解决轨迹整体形状问题；高频数值上升并不等价于更准确，结合端点和形状误差，不能将其解释为稳定的高频恢复。

预测 goal 条件下，A 的 endpoint translation error 为 `0.03061 m`，B 为 `0.03602 m`。频谱报告与图片位于：

* A：[spectrum_gt](file:///home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/spectrum_gt/spectrum_comparison.png)
* B：[spectrum_gt](file:///home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/spectrum_gt/spectrum_comparison.png)

## 6. 可视化资产

两组使用同一个 `visualize_training_inference.py`，对 test split 的全部 18 个 episode 生成 GT 与 top-1 goal/trajectory 对照图、goal 对照图和汇总图。每组还使用 `visualize_motion_fields.py` 对 `test/episode_14` 输出场图：

* A 运动场：[fields_visuals/test_episode_14.png](file:///home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/fields_visuals/test_episode_14.png)
* B 运动场：[fields_visuals/test_episode_14.png](file:///home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/fields_visuals/test_episode_14.png)
* A 轨迹汇总：[test_inference_gt_vs_top1_summary.png](file:///home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/test_visualization/test_inference_gt_vs_top1_summary.png)
* B 轨迹汇总：[test_inference_gt_vs_top1_summary.png](file:///home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/test_visualization/test_inference_gt_vs_top1_summary.png)

每个 episode 还保存单独的 `test_episode_<id>_gt_vs_top1.png` 和 `test_episode_<id>_goal_gt_vs_top1.png`，便于后续挑选论文图例。

## 7. 当前结论与后续使用建议

1. **功能运动场确实可学习且对输出有因果影响。** A/B 的 uniform、rolled 干预均造成退化，且 learned 场具有非均匀 entropy/peak 统计。
2. **Goal-only 不是当前默认方案。** 它更偏向终态旋转，然而整体平移和轨迹指标下降，说明仅让终态损失塑造场会丢失对中间运动有用的区域信息。
3. **论文/主线保留 A Joint。** A 作为完整模型，B 作为“field supervision source”消融，清楚展示轨迹损失对功能场的补充作用；不要把 B 的单项旋转改善写成全面提升。
4. 后续若要提升场的可视化显著度，应优先改进数据覆盖、场监督或独立的轻量场校准，而不是直接增加人为几何模块。本次实验不改变后续 trajectory transformer，因而可以安全回退并作为后续实验基线。

## 8. 可复现实验命令

```bash
# B training (A uses the complete v2_joint checkpoint above)
python scripts/stage2/train_motion_functional_field.py \
  --config configs/stage2/motion_field_v2_pouring_lfv_goal_only_field.yaml

# test metrics
python scripts/stage2/evaluate_motion_functional_field.py \
  --checkpoint /home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/checkpoints/best.pt \
  --cache-root /home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1 --split test \
  --device cpu --num-goals 4 --num-trajectories 1 --use-ema

# field and trajectory visualizations
python scripts/stage2/visualize_motion_fields.py \
  --checkpoint /home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/checkpoints/best.pt \
  --cache-root /home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1 --split test \
  --episode-id episode_14 --output /home/users1/ljian/lfv_runs/stage2/motion_field_ablation/goal_only_field/fields_visuals/test_episode_14 --device cpu
```

## 9. 版本与回退

* `stage2-goal-field-ablation-pre-v1`：梯度路由修改前的代码基线；
* `stage2-goal-field-ablation-v1`：A/B 配置、实现和单元测试；
* 本实验记录提交后新增结果 tag：`stage2-goal-field-ablation-results-v1`。

所有 checkpoint 和可视化输出保存在 `/home/users1/ljian/lfv_runs`，没有写入原始数据集。
