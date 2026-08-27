# Goal 与 Motion Functional Field 改进实验记录

本文档记录在 `pouring_lfv` 上对第二阶段 Goal 分支和功能运动场的低复杂度改进。Trajectory Diffusion 的网络和损失没有改动；所有新增分支均可由配置开关关闭，旧的 Joint checkpoint 仍可加载。

## 1. 原始结构与改进动机

原模型通过两个 PointNet、双向 Cross-Attention 和 relevance head 得到 manipulated/reference Motion Field，再将场加权特征池化成三个 scene token。Goal Pose Diffusion 只接收这三个 token，并回归 9D 终态位姿。这个路径能够让场影响 Goal，但没有显式告诉 Goal decoder“被操作物体功能区域在哪里、参考区域在哪里以及两者的相对几何关系”，所以场容易保持较高熵，终态平移也可能偏离可执行区域。

本次新增两项：

1. **Field-derived functional relation tokens**：使用 Motion Field 对输入点坐标做可微加权，生成 manipulated/reference anchor XYZ；再将 anchor 与双向关系 token 融合为 3 个 Goal-only relation tokens。它们不是人工定义的 OBB、PCA 或固定中心，而是网络场的连续矩。
2. **可选 Goal candidate scorer**：对 K 个扩散 Goal 候选使用一个很小的 MLP 评分器；训练时用 GT pose 为正样本、归一化 pose 加随机扰动为负样本，推理时按 score 排序后再交给 Trajectory Diffusion。默认旧配置关闭该分支。

另外实现了可选 Sparsemax field normalization。它能产生精确零值、便于观察，但不默认使用，因为直接替换 Softmax 会损害 pose 精度。

## 2. 代码接口

* `ContextEncoding.goal_relation_tokens`：`[B,3,H]`，只传给 Goal decoder；Trajectory 仍使用原来的 `encoding.tokens` (`[B,3,H]`)。
* `ContextEncoding.manipulated_anchor_xyz/reference_anchor_xyz`：`[B,3]`，用于调试和可视化。
* `GoalPoseDecoder.forward(..., goal_relation_tokens=None)`：有条件地拼接 Goal relation memory。
* `GoalPoseDiffuser.score_candidates(...)`：返回 `[B,K]` 候选 logits；`ThreeTokenHierarchicalDiffusion.sample` 会在 scorer 开启时排序 Goals，并同步更新 `goal_ids`。
* `model.goal_relation_conditioning`、`model.goal_candidate_scoring`、`model.motion_field_normalization` 均为配置项，默认值保持旧行为。

## 3. 实验配置与 checkpoint

共同数据缓存为 `/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1`，测试 split 固定 18 个 episode，seed=42，EMA，K=4，CPU 推理。

| 实验 | 配置 | checkpoint | 说明 |
|---|---|---|---|
| Joint baseline | `motion_field_v2_pouring_lfv.yaml` | `/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/checkpoints/best.pt` | 原始完整模型，epoch 129 |
| Anchor（温度/关系权重改动） | `motion_field_v3_goal_anchor.yaml` | `/home/users1/ljian/lfv_runs/stage2/goal_anchor_v1/checkpoints/best.pt` | 约 65 epoch，非默认 |
| Anchor baseline | `motion_field_v3_goal_anchor_baseline.yaml` | `/home/users1/ljian/lfv_runs/stage2/goal_anchor_baseline_v1/checkpoints/best.pt` | 复用原 Softmax、temperature=0.25、pair weight=0.25；best epoch 77 |
| Anchor + scorer | `motion_field_v3_goal_anchor_score.yaml` | `/home/users1/ljian/lfv_runs/stage2/goal_anchor_score_v1/checkpoints/best.pt` | 约 61 epoch，候选评分消融 |
| Gated anchor | 使用 anchor baseline 加 residual gate | `/home/users1/ljian/lfv_runs/stage2/goal_anchor_gated_v1/checkpoints/best.pt` | 约 57 epoch，验证小门控是否保护旧模型 |

短训练结果用于结构筛选；未达到原始 A 模型的完整 epoch 129，因此不应将其写成最终 SOTA 结果。

## 4. 测试结果（EMA、K=4）

| 指标 | Joint baseline | Anchor | Anchor baseline | Anchor + scorer | Gated anchor |
|---|---:|---:|---:|---:|---:|
| Goal 平移误差 (m) | **0.02925** | 0.03605 | 0.03447 | 0.03832 | 0.04445 |
| Goal 旋转误差 (deg) | 25.72 | **22.55** | **22.08** | 24.01 | 23.68 |
| Trajectory 平移误差 (m) | **0.04288** | 0.04681 | 0.04462 | 0.05321 | 0.05772 |
| Trajectory endpoint 平移误差 (m) | **0.03041** | 0.04051 | 0.03570 | 0.04420 | 0.05869 |
| Trajectory 旋转误差 (deg) | **14.04** | 14.84 | 14.30 | 15.12 | 16.19 |

结果说明：Functional relation tokens 对终态旋转有一致的改善趋势，但尚未改善平移和轨迹端点。当前 scorer 的训练负样本较粗糙，排序后反而恶化，因此只保留为可选接口，不作为默认策略。Anchor baseline 相比改变 temperature/pair weight 的版本更稳定，但仍没有超过 Joint baseline。

## 5. Motion Field 可视化与稀疏化实验

Anchor baseline 的测试场图为：

`/home/users1/ljian/lfv_runs/stage2/goal_anchor_baseline_v1/fields_visuals/test_episode_14.png`

测试统计：manipulated entropy `0.9936`、peak mass `0.00623`。原始 Joint baseline 在相同 episode 上 entropy 约 `0.9305`、peak mass `0.0201`，说明 anchor 本身没有自动使场变得更集中；它只是将现有场转化为可供 Goal 使用的几何关系。

在原 Joint checkpoint 上仅将 Softmax 替换为 Sparsemax，测试场 entropy 降至约 `0.186`、peak mass 升至约 `0.598`，但 Goal 平移误差从 `0.0293 m` 恶化到 `0.0400 m`，轨迹 endpoint 平移误差从约 `0.0304 m` 恶化到约 `0.0415 m`。这表明“视觉上更尖”不等价于“任务上更正确”，因此 Sparsemax 只作为诊断选项。

Anchor baseline 的轨迹可视化汇总为：

`/home/users1/ljian/lfv_runs/stage2/goal_anchor_baseline_v1/test_visualization/test_inference_gt_vs_top1_summary.png`

## 6. 当前推荐

当前默认仍使用原始 Joint baseline。推荐的最小后续方向不是继续降低 temperature 或强行做 entropy 塌缩，而是：

1. 保留 Field-derived relation tokens，但在完整 epoch、同一 checkpoint 选择协议下重新训练；
2. 用任务容忍度构造更合理的 Goal candidate ranking target，而不是只用随机 pose 作为负样本；
3. 对杯口这种多解任务报告功能关系成功率（杯口方向/碗中心关系）和 best-of-K，而不只报告单一 GT pose 的点误差；
4. 暂不改变 Trajectory Diffusion，先确认 Goal anchor 是否真正改善终态，再把改进后的 Goal 传入轨迹分支。

## 7. 版本与回退

* `stage2-goal-anchor-pretraining-v1`：anchor/scorer 初始实现；
* `stage2-goal-anchor-gated-pretraining-v1`：加入 gate 与 Sparsemax 选项后的版本；
* 当前代码分支：`stage2/motion-functional-field`。

若需要恢复原始实现，可回到 `stage2-joint-baseline-restored-v1`；若需要恢复本次改进代码，可使用 `stage2-goal-anchor-gated-pretraining-v1`。
