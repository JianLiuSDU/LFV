# Stage 2 Motion Functional Field 逐版实验记录

> 范围：仅修改 Stage 2 模型及其训练、评估和可视化基础设施；不包含实机、AUBO、
> SAM3D、抓取或跨实例场迁移。本文件中的状态按实际完成情况更新，不把计划写成结果。

## 固定数据与协议

- 源数据：`/media/ljian/lj/data_3d/pouring_lfv`；
- 固定缓存：`/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1`；
- 有效 episode：179；train/val/test：143/18/18；seed：42；
- 每个角色 256 个逐点对齐 XYZ–DINO 样本，DINO 维度 384；
- Full64 累积 Pose9D 轨迹；
- 当前缺少可靠 `object_instance_id`，因此测试只称为 `episode_split_baseline`。

固定文件哈希：

```text
manifest.json       1b849b2b0c45a7128fb27b6f04caf25a657f41fb19971916d761a05c962daa12
split_manifest.json 0a2e09088b828c8bdebdbcd49214706166de212f45ac47075785f119bd56f7e8
```

## V0：A3b 基线

- Git tag：`stage2-motion-field-v0-a3b`；
- Encoder：两个 PointNet、双向 cross-attention、三个 max/global pooled tokens；
- Goal：4层 Goal Pose Diffusion；
- Trajectory：A3b 6层 Transformer、离散时间编码、门控 Goal Context Mixer、4个门控
  Phase Tokens；
- 当前 attention summary 只是调试统计，不参与 token 聚合，因此不是显式 Motion
  Functional Field。

固定 test 结果：

| 指标 | V0 |
|---|---:|
| Goal top-1 translation | 0.02874 m |
| Goal top-1 rotation | 24.42 deg |
| Trajectory top-1 translation | 0.04290 m |
| Trajectory top-1 rotation | 14.00 deg |
| GT-goal low position retention / cosine | 0.9220 / 0.8926 |
| GT-goal mid position retention / cosine | 0.3243 / 0.3786 |
| GT-goal mid velocity retention / cosine | 0.2747 / 0.3232 |
| GT-goal dominant-curvature frame error | 6.22 frames |

## V1：Independent Explicit Motion Fields

状态：代码、单元测试和32条样本overfit已完成；判定为结构性负向结果，不进入完整
179条数据训练。

V1 不改变两个 Diffusion Decoder，只替换两个 Directional Relation 的池化方式。对
manipulated 查询方向：

\[
U_m(i)=\operatorname{Fuse}
\left(X_m(i),\operatorname{CA}_{m\leftarrow r}(X_m,X_r)_i\right),
\]

\[
H_m(i)=\operatorname{Softmax}_i
\left(h_m(U_m(i))/\tau\right),
\qquad
z_{m\leftarrow r}=\sum_iH_m(i)U_m(i).
\]

参考物体方向同理得到 \(H_r\) 和 \(z_{r\leftarrow m}\)。\(H_m,H_r\) 是模型正常
forward 中必定计算并用于预测的张量，不是由 attention 事后平均产生的解释图。V1
仍保留原始 global token，用来隔离检验“最小显式场”是否足够；若模型通过该 token
绕过功能场，则进入 V2 联合关系场瓶颈。

新增输出：

```text
manipulated_motion_field [B,Nm]
reference_motion_field   [B,Nr]
manipulated_motion_logits [B,Nm]
reference_motion_logits   [B,Nr]
```

新增训练诊断：归一化场熵与峰值概率质量。新增可视化同时输出首帧 RGB 叠加、两组
三维点云热力图和场值分布。

已完成测试：

```text
20 passed
```

覆盖旧模式兼容、场 shape/归一化、点顺序置换等变、三个 token 置换不变、Goal 与
Trajectory 联合损失到两个 relevance heads 的非零梯度，以及 checkpoint 恢复与采样
复现。

首次overfit训练在恢复checkpoint时发现旧EMA实现会直接采用checkpoint tensor的设备，
从而可能让恢复后的CPU shadow与GPU模型冲突。V1.1将EMA状态显式转换到当前模型参数
的device和dtype；该修改只修复断点续训，不改变模型前向或损失。

V1 overfit运行至epoch 159，最终train/同集val total分别为0.07762/0.07921，证明
新的加权聚合能够正常训练并高度拟合32条运动。但最终归一化场熵仍为
0.99876/0.99875，平均peak mass约0.00501，而256点均匀分布为0.00391。episode 0与
episode 12的RGB/3D可视化也显示以全物体缓慢梯度为主，而不是明确局部功能区域。

因此V1的低任务损失不能视为Motion Functional Field成功。结论是：独立head与关系
加权本身不足；原始global token和平均关系表示足以拟合任务，使模型没有动力形成明确
重要性。V1被保留为必要负向消融。

本地证据：

```text
/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v1_overfit32
```

## V2：Joint Functional Relation Bottleneck

状态：代码、单元测试、32条样本 overfit、179条样本固定划分训练、测试和因果消融均已完成。

V2继续只修改Encoder。两个方向的逐点logits与双向attention的互惠兼容性共同构造：

\[
L_{ij}=a_i+b_j+\lambda_p\frac{1}{2}
\left(\log A_{m\rightarrow r,ij}+\log A_{r\rightarrow m,ji}\right),
\]

\[
J_{ij}=\operatorname{Softmax}_{i,j}(L_{ij}/\tau).
\]

两个可视化场严格定义为联合关系分布的边缘：

\[
H_m(i)=\sum_jJ_{ij},\qquad H_r(j)=\sum_iJ_{ij}.
\]

两个方向relation tokens使用对应边缘场加权；原始未筛选global token被替换为两个
field-weighted对象摘要的融合。因而三个Decoder context tokens全部依赖 \(J\)，不再
保留与其同容量的直接旁路。Goal和A3b Trajectory网络完全不变。

V2第一轮固定 `temperature=0.25`、`pair_weight=0.25`。温度只控制连续竞争强度，
不会加入杯口、碗口、PCA、法向或其他人工任务特征。

### V2 训练结果

V2 overfit 运行至 epoch 159，最佳训练/同集验证 total 约为 0.07538/0.08108；
归一化 manipulated/reference field 熵约为 0.876/0.900，peak mass 约为 0.0397/0.026，
明显区别于 256 点均匀分布（peak=0.00391）。在固定的 143/18/18 episode 划分上，
全量训练在 epoch 209 早停，最佳 checkpoint 为 epoch 129，验证 total=0.47085。

CPU 固定测试（test=18 episodes，EMA 权重，K=4 goals、每个 goal 1 条 trajectory）为：

| 指标 | V0 A3b | V2 Joint Field |
|---|---:|---:|
| Goal top-1 translation | 0.02874 m | 0.02925 m |
| Goal top-1 rotation | 24.42 deg | 25.72 deg |
| Trajectory top-1 translation | 0.04290 m | 0.04288 m |
| Trajectory top-1 rotation | 14.00 deg | 14.04 deg |
| Trajectory first-step translation error | — | 0.00267 m |

因此，加入显式场瓶颈没有破坏原有轨迹任务性能；goal 旋转存在约 1 度量级的波动，
需要在后续更高 K 的 GPU 评估中复核，不能据此宣称提升。

### 因果性与可视化证据

在同一测试协议中，把正常的 (H_m,H_r) 替换成均匀场，或在点维度循环打乱场，
会改变三个 context tokens 并造成性能退化：

| 场输入 | Goal top-1 translation | Trajectory top-1 translation | Trajectory rotation |
|---|---:|---:|---:|
| learned field | 0.02932 m | 0.04198 m | 13.77 deg |
| uniform field | 0.03302 m | 0.04357 m | 14.34 deg |
| rolled field | 0.03305 m | 0.04355 m | 14.36 deg |

这说明运动场参与了预测，而不是仅作为事后 attention 可视化。需要保持证据边界：
因果退化证明的是场的任务相关性，不等于已经证明场在所有实例上等价于真实的人类
运动功能区域。

固定可视化样本保存在：

```text
/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/fields_visuals/
  train_episode_0.png
  val_episode_15.png
  test_episode_14.png
```

episode 0 的 manipulated 场在杯体局部形成明显峰值，reference 场集中于碗内/碗缘
区域；val episode 15 仍能观察到杯体局部峰值。test episode 14 的场相对平坦，说明
仅靠无额外场监督的任务损失，场的跨 episode 稳定性仍有限；该样本也被保留作为诚实的
失败/不确定性证据，而不是挑选性展示。

频谱评估（test=18，GT goal）为 position low-energy retention=0.887、mid=0.339、
high=0.195；velocity low/mid/high retention=0.799/0.245/0.151。与 V0 的低/中频
指标同量级，V2 当前主要解决“场是否参与运动预测”，尚未解决轨迹高频细节被过度平滑
的问题。

## 后续进入完整训练的门槛

1. 32条训练样本 overfit 时 task loss 明显下降；
2. 两个场不是 NaN、完全均匀或单点塌缩；
3. 均匀/打乱场干预会改变 Context Tokens 和运动预测；
4. 短训练 validation 指标没有明显退化；
5. 完整训练后低频和中频指标不能用无相位伪高频替代；
6. 如果V1不能形成稳定功能区域，保留该负向结果并进入V2，而不是通过人工杯口特征
   强迫得到漂亮热力图。

V2 已通过前四项并完成固定测试。下一轮若要提高场的跨 episode 稳定性，应优先考虑
训练内的轻量一致性或数据增强，并保持同一 encoder/decoder 接口；不得直接加入手工
杯口检测、PCA 或后处理热力图来替代网络学习。当前版本的可复现实验入口是：

```text
branch: stage2/motion-functional-field
tag:    stage2-motion-field-v2.1-causal-eval
best:   /home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/checkpoints/best.pt
```
