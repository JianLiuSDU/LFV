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

状态：代码和单元测试已完成；训练尚未开始。

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

## 后续进入完整训练的门槛

1. 32条训练样本 overfit 时 task loss 明显下降；
2. 两个场不是 NaN、完全均匀或单点塌缩；
3. 均匀/打乱场干预会改变 Context Tokens 和运动预测；
4. 短训练 validation 指标没有明显退化；
5. 完整训练后低频和中频指标不能用无相位伪高频替代；
6. 如果V1不能形成稳定功能区域，保留该负向结果并进入V2，而不是通过人工杯口特征
   强迫得到漂亮热力图。
