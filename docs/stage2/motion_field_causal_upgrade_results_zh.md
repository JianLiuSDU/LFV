# Motion Functional Field 改造：实现、实验与当前结论

本文档记录 `motion_field_cross_instance_upgrade_plan_zh.md` 的实际执行结果。它区分
“已经写入代码并完成验证”和“仍需要多实例数据才能验证”的内容，避免把当前的单源实验
误写成跨实例成功。

## 1. 本次改造的目标

原始 relevance head 产生的热力图可能只是 cross-attention 的可视化，而不是预测真正
依赖的任务信息。本次改造固定了三条约束：

1. **Field bottleneck**：Goal/Trajectory decoder 只接收经过运动场加权的三个场景
   token；不再存在 `manipulated_global/reference_global -> decoder` 的旁路。
2. **Task-effect probe**：在相同随机噪声和相同输入下，将 learned field 替换为
   uniform、循环移位或去掉最高质量的 `drop_top` field，并比较去噪误差。
3. **Consistency and transfer**：同组样本的场用 DINO 相似度进行软传输并计算对称
   KL；推理时把源 memory 的场用 FGW 传到目标，再以置信度调节先验融合权重。

这三点不增加新的视觉 backbone，也不引入 PCA、OBB、手工杯口坐标或 KNN 几何特征。

## 2. 已实现的计算路径

### 2.1 在线运动场

两个 PointNet 分别编码操作物体和参考物体的 XYZ–DINO 特征。双向 cross-attention
产生逐点关系特征 (h_i^m,h_j^r) 以及 logits (l_i^m,l_j^r)。joint 模式将双向
logits 和 attention compatibility 合成

[
 L_{ij}=l_i^m+l_j^r+lambda_plog A_{ij},qquad
 R_{ij}=operatorname{softmax}(L_{ij}/T),
]

并将 (R) 的边缘化结果作为两个物体的归一化运动场：

[
 r_i^m=sum_jR_{ij},qquad r_j^r=sum_iR_{ij}.
]

`motion_field_power` 在 log-space 中执行，以避免 (N	imes N) 联合分布做幂次时
下溢：

[
 operatorname{sharpen}(R)=
 operatorname{softmax}(plog(R+epsilon)).
]

三个 token 的场加权形式为

[
 c_m=sum_i r_i^m h_i^m,quad
 c_r=sum_j r_j^r h_j^r,quad
 c_0=operatorname{MLP}([c_m,c_r]).
]

Goal 和 Trajectory 两个扩散分支只读取 ([c_0,c_m,c_r])。因此，替换场会真正改变
扩散条件，而不只是改变一张用于展示的热力图。

### 2.2 反事实任务作用约束

令 (L_	ext{learned})、(L_	ext{uniform}) 和 (L_	ext{drop}) 分别是使用三种
场得到的 Goal/Trajectory 去噪损失。训练中加入

[
 L_	ext{cause}=ig[m+L_	ext{learned}
                    -operatorname{sg}(L_	ext{uniform})ig]_+,
]

以及可反传的峰值移除项

[
 L_	ext{peak}=ig[m+L_	ext{learned}-L_	ext{drop}ig]_+.
]

其中 `sg` 只用于构造稳定的 ranking target，`drop_top` 分支保持可微，使 relevance
head 能收到“高质量区域被移除后预测变差”的梯度。

### 2.3 同演示一致性

`field_consistency_group` 相同的样本通过 DINO affinity 形成软传输矩阵，比较传输后
的两个归一化场的对称 KL。当前缓存的原始 `object_instance_id` 为空，因此配置中的
`pouring_source_instance` fallback 将所有 pouring episode 视作同一源组；这一点是数据
质量限制，不能当成真正的跨实例监督。

### 2.4 Prior–Evidence Fusion

`build_motion_field_memory.py` 将一个源 episode 的在线场、DINO 和点云保存为 memory。
`evaluate_motion_field_transfer.py` 使用同一 FGW 传输算子得到目标 prior，并评估：

- `direct`：只使用当前场（evidence-only）；
- `transfer_only`：只使用 transported prior；
- `full`：当前场和 prior 的 confidence-aware 融合。

融合前先计算两个场的 Jensen–Shannon agreement。当前实现保留算术混合以兼容旧模型，
但当 agreement 低时会降低有效 prior weight；场形状和权重均写入输出 JSON。

## 3. 版本与回退点

所有网络变更均在 `stage2/motion-functional-field` 分支提交并推送到 GitHub，提交链为：

| 版本 | 提交 | 内容 |
|---|---|---|
| P0 | `dc209d3` | 改造计划和验收指标 |
| v1 | `9cf44da` | causal field bottleneck、confidence fusion |
| v2 | `359a8c3` | FGW transfer、selective probe 配置 |
| v3 | `d0c39f5` | `drop_top` 反事实测试 |
| v4 | `839892c` | field sharpening |
| v5 | `9b807c5` | balanced 配置 |
| v5-fix | `6344f53` | log-space power，修复联合场数值下溢 |
| v6 | `9163cdd` | 80 epoch long-run 配置 |

需要回退时可在确认工作区没有待提交修改后执行 `git switch --detach <commit>`，或从
对应提交新建实验分支。部署包 `deployment_bundle/aubo_camera_execution.zip` 未被改动。

## 4. 训练结果：v6 balanced long

训练配置为 `configs/stage2/motion_field_v6_balanced_pouring_lfv_long.yaml`：

- 数据：`/media/ljian/lj/data_3d/pouring_lfv` 的缓存
  `/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1`；179 个 episode，
  训练/验证/测试为 143/18/18；每个物体 256 点，DINO 384 维；
- hidden dim 128，4 heads，joint field，temperature 0.10，power 1.5；
- causal/drop/consistency 权重 0.10/0.10/0.02；80 epochs，EMA；
- 最佳 checkpoint（epoch 65）：
  `/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v6_balanced_pouring_lfv_long/checkpoints/best.pt`。

训练总损失从 epoch 0 的 4.773 降到 epoch 79 的 0.381；最佳验证总损失为 0.52054。
Goal 和 Trajectory 的分项损失同步下降，没有出现轨迹项被单独牺牲的情况。

### 4.1 Test field intervention

报告文件：
`/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v6_balanced_pouring_lfv_long/causality_test.json`。

18 个 test episode、每个输入 4 个 goal sample 的均值如下：

| 场条件 | Goal top-1 平移误差 | Trajectory top-1 平移误差 | 操作物体场熵 | 操作物体峰值 |
|---|---:|---:|---:|---:|
| learned | 51.89 mm | 70.97 mm | 0.9915 | 0.00672 |
| uniform | 54.26 mm | 71.10 mm | 1.0000 | 0.00391 |
| rolled | 54.09 mm | 71.17 mm | 0.9915 | 0.00672 |
| drop_top | 53.26 mm | 70.92 mm | 0.9960 | 0.00500 |

因此 learned 场相对 uniform 使 Goal 平移误差下降约 2.37 mm，相对循环移位下降约
2.20 mm；`drop_top` 的退化约 1.37 mm。Trajectory 的差异仅约 0.1–0.2 mm，说明
当前场的可测作用主要落在 Goal 分支，尚不足以声称对整条轨迹具有强因果必要性。
旋转误差也没有稳定改善，应在论文中如实报告。

场熵仍接近 1，说明这是“中等集中、非单点塌缩”的关系场，而不是非常尖锐的杯口
掩码。可视化文件：
`/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v6_balanced_pouring_lfv_long/episode_102_motion_field.png`。

## 5. FGW 迁移/融合结果

源 memory 为 training episode 0，测试仍是同一缓存中的 18 个 episode。结果文件：
`/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v6_balanced_pouring_lfv_long/transfer_test.json`。

| 条件 | Goal top-1 平移误差 | Goal best 平移误差 | Trajectory top-1 平移误差 |
|---|---:|---:|---:|
| direct | 51.54 mm | 51.12 mm | 71.08 mm |
| transfer_only | 55.17 mm | 54.83 mm | 72.32 mm |
| full | 53.14 mm | 52.76 mm | 71.63 mm |

平均 FGW confidence 约 0.0133，属于低置信度迁移。`transfer_only` 比 direct 差约
3.64 mm，而 confidence-aware `full` 将损失差缩小到约 1.60 mm。这验证了“不能直接
搬运 source field、需要 prior–evidence fusion”的必要性，但还没有验证“迁移能提升
未见物体实例”。原因是当前缓存的 `object_instance_id` 全为空，且 memory 与测试数据
来自同一 pouring 物体分布；这不是严格的 cross-instance benchmark。

## 6. 仿真快速检查

使用已有蓝色杯子 snapshot 运行：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/stage2/infer_sim_snapshot.py \
  --checkpoint /home/users1/ljian/lfv_runs/stage2/motion_functional_field/v6_balanced_pouring_lfv_long/checkpoints/best.pt \
  --snapshot /home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/blue_mug_seed_0/snapshot/pouring_snapshot.npz \
  --cache-root /home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1 \
  --output-dir /home/users1/ljian/lfv_runs/stage2/motion_functional_field/v6_balanced_pouring_lfv_long/sim_blue_mug \
  --device cpu --num-goals 8 --num-trajectories 2
```

固定输出包括 `functional_motion_prediction.npz`、`goal_pose_candidates_overlay.png`、
`full64_coordinate_frames_overlay.png`、`encoder_cross_attention_summary.png` 和
`simulation_inference_summary.png`。这次检查暴露出明显 domain gap：被选轨迹首步约
14.98 mm，而训练集首步 p95 约 1.49 mm；预测终态到可见碗中心约 0.214 m。因此该模型
checkpoint 当前只能作为网络机制的验证，不能直接宣称已完成仿真任务成功。

## 7. 回归测试与可复现命令

Stage 2 单元测试结果：`37 passed in 3.10s`。

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python -m pytest -q tests/stage2
```

重新生成三个核心报告：

```bash
# field causality
python scripts/stage2/evaluate_motion_field_causality.py ...
# source memory
python scripts/stage2/build_motion_field_memory.py ...
# direct / transfer-only / full
python scripts/stage2/evaluate_motion_field_transfer.py ...
```

完整参数以本次输出 JSON 中记录的绝对路径为准；所有脚本都固定 seed，并使用 EMA
checkpoint。

## 8. 尚未完成、不可提前声称的部分

1. 需要修复数据缓存中的 `object_instance_id`，按真实杯子资产划分 train/val/test，
   再运行至少四个杯子资产的 cross-instance 测试；
2. 需要从仿真 mesh 提取杯口 rim 与碗口 opening，加入 ROF（rim-over-opening fraction）、
   碰撞率和任务成功率，而不是只报告杯身中心距离；
3. 需要比较多个源 memory、错误/打乱 prior 以及 prior dropout，验证 gate 是否会拒绝
   低质量先验；
4. 需要针对 Trajectory 分支增加不依赖手工几何的路径级 field-effect 指标，否则当前
   结果只能支持“field 对 Goal 有可测影响”，不能支持“field 对整条轨迹是必要的”。

这些是下一轮实验门槛，不应通过改变测试阈值或选择性展示可视化来掩盖。

