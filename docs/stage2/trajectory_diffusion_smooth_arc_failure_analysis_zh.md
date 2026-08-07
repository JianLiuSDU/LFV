# Stage 2 轨迹扩散的“平滑弧线”与首段突变问题分析

> 日期：2026-08-07
> 分析对象：`full_joint_start_fixed_v3` 及其对应的三 Token Encoder、Goal Diffusion、Trajectory Diffusion Transformer
> 本文只做代码、标签、训练结果与旧版 `object_centric_diffusion` 的诊断对比，不修改模型。

> **后续状态说明：** 本文分析的是修复前的 `full_joint_start_fixed_v3` checkpoint。轨迹位置编码现已改为离散帧索引，并对缺少版本字段的旧 checkpoint 保留 legacy 行为；新的共享 Encoder 与阶段感知 Transformer 设计见 [joint_encoder_and_stage_aware_trajectory_transformer_zh.md](joint_encoder_and_stage_aware_trajectory_transformer_zh.md)。

## 1. 结论

当前现象不是“终态位姿没有学好”，也不主要是推理时 Goal 采样误差造成的。终态是一个低维、静态的物体关系问题，当前三个场景 Token 足以支持它；但完整轨迹要求网络辨认时间位置、动作阶段和局部几何约束。当前 Trajectory Diffusion 在这三方面都偏弱，因而把不同示范中的“先抬起、再搬运、最后倾倒”压缩成一条低频、平滑的折中弧线。首段突变则是同一问题在起点边界处的表现：固定起点只是 Transformer 中的一个上下文 Token，真正生成的第 1 帧没有在每次扩散迭代中被硬约束，而弱时间编码和全局注意力会把它直接拉向整条平滑桥接曲线。

最关键的原因按优先级排列如下：

1. **当前轨迹位置编码几乎退化。** 64 个进度值被缩放到 `[0,1]`，再送入最高频率仅为 `1 rad` 的标准正弦公式。实测 128 维进度编码的有效秩只有约 `1.20`；若输入帧索引 `0…63`，有效秩约为 `16.90`。全局自注意力因此很难清楚区分第 8、24、40、56 帧及其动作阶段。
2. **当前 Transformer 缺少旧版 1D U-Net 的强时间归纳偏置。** 每层只有一次 `kernel=3` 的同尺度卷积，随后立即对全部 64 帧做无掩码全局自注意力；旧版使用多尺度 1D 卷积、下采样、上采样和 Skip Connection，更容易保留“局部运动—阶段转折—全局轨迹”的层级结构。
3. **当前不是严格的起终点轨迹补全。** 旧版在训练和每一步 DDIM 去噪中都把第 0 帧与第 63 帧硬写回，网络只需生成中间 62 帧；当前只把起点作为固定注意力 Token，终点仍是软条件和软损失，网络实际承担的是“生成全部 63 帧并尽量靠近 Goal”。这更容易收敛为一条平滑桥。
4. **三个场景 Token 对终态足够，但对中间路径过度压缩。** Encoder 最终只向轨迹网络提供 `initial`、`manipulated→reference`、`reference→manipulated` 三个全局 Token；每个轨迹帧再交叉注意这三个 Token 和一个 Goal Token。局部点云几何、净空、把持物体在各阶段相对参考物体的位置都已丢失，所以网络知道“最后去哪里”，却缺少“中间应该怎样去”的信息。
5. **逐帧 `x0` 回归、阶段未对齐和低采样多样性共同产生条件均值。** 训练标签虽都是 64 帧，但按运动弧长重采样，不同演示中的抬升结束、水平搬运开始和倾倒开始并未语义对齐。逐帧 MSE/L1 在这种条件下倾向输出低频的条件均值/中位数，而不是保持某一条示范中的明显折点。

所以，用户观察到“终态很好、轨迹却像统一的平滑弧线”是符合当前结构的：Goal Decoder 解决的是一个低维静态映射，Trajectory Decoder 却被要求在弱时间编码、强上下文压缩和软边界下恢复一个阶段化的 64 帧分布。

## 2. 当前标签和模型实际在学习什么

### 2.1 轨迹不是相邻帧位移，而是累计位姿

当前缓存中的：

```text
trajectory_pose9d[B, 64, 9]
```

每一帧是从初始时刻到当前时刻的累计刚体变换：

```text
T_0_to_k,  k = 0 ... 63
```

Pose9D 为：

```text
[tx, ty, tz, rotation_6d]
```

其中第 0 帧为单位变换，第 63 帧就是 `goal_pose9d`。缓存构建从 `T_matrices_4x4` 读取累计变换，经场景原点和平移尺度变换后直接转换成 Pose9D；并没有把 `T_{k-1}^{-1}T_k` 当作标签。因此当前表示方式本身没有“把相邻位移错误地不断累加”的问题。

累计位姿有利于终态收敛，但它也使每一个时间位置都变成一次绝对/累计回归。只要时间位置区分弱、阶段没有对齐，网络就很容易在起终点之间学习一条平滑的低频函数。

### 2.2 当前 Trajectory Decoder

当前网络对第 1 至第 63 帧的带噪 Pose9D 做 MLP 投影，并在序列最前方追加一个固定的起点 Token：

```text
63 noisy pose tokens
    + one fixed start token
    + progress embedding
    ↓
6 × [temporal Conv1d(k=3)
     + full self-attention
     + cross-attention(scene3 + goal1)
     + FFN]
    ↓
63 clean Pose9D predictions
```

它是非因果模型，这一点本身合理，因为整条轨迹扩散需要同时利用过去和未来；问题不在于“非因果”，而在于：

- 全局注意力占主导，却缺少足够可分辨的帧位置编码；
- 局部卷积只有单一尺度；
- 起点 Token 会在每层后被写回，但第 1 个预测帧不会被写回；
- Goal 只是 Memory Token，最终帧没有被扩散调度器硬钳制；
- 三个场景 Token 没有向每个时间位置提供阶段相关的局部几何。

### 2.3 当前损失

当前调度器使用 `prediction_type="sample"`，即网络直接预测干净轨迹 `x0`。主要损失可以概括为：

```text
L = L_x0
  + L_translation
  + 0.5 L_SO3
  + 0.5 L_velocity
  + 0.1 L_acceleration
  + 2.0 L_start
  + 1.0 L_endpoint
```

另外，`L_x0` 中第一个生成帧的权重为 `20`，最后一帧权重为 `2`。

这里要准确区分两件事：`L_velocity` 和 `L_acceleration` 是预测导数与 GT 导数的匹配，并不是直接把速度或加速度压到零，所以它们不是平滑弧线的唯一原因。但是，当训练样本的阶段发生时间没有对齐时，L1 导数匹配仍倾向于选择跨样本的低频中位数。仿真候选排序又显式加入了：

```text
+ 30 × mean_second_difference
```

因此仿真端会在已经相似的候选里进一步偏好更平滑的曲线。训练集可视化直接使用未排序的第一个样本，也同样出现弧线，所以候选排序只是仿真端的放大因素，不是训练退化的根因。

## 3. 证据

### 3.1 训练集上已出现退化，不只是仿真域差异

当前训练集可视化中，GT 普遍包含比较明确的“先抬升，再横移/倾倒”阶段，而预测结果更接近一条连续圆弧。四个训练样本的 Top-1 全程平移误差为：

| Episode | Top-1 平均平移误差 | Top-1 终点误差 |
|---|---:|---:|
| episode_152 | 2.33 cm | 0.98 cm |
| episode_90 | 2.99 cm | 1.22 cm |
| episode_33 | 3.19 cm | 2.04 cm |
| episode_12 | 3.60 cm | 2.61 cm |

这说明问题在训练分布内部已经存在，不能只解释为蓝色杯子仿真场景的域偏移。

### 3.2 把真实 Goal 直接输入，路径仍然不对

为了隔离 Goal Diffusion 的误差，使用相同 EMA checkpoint，将上述四个训练样本的 GT Goal 直接输入 Trajectory Diffusion，每个样本采样 4 条轨迹。结果为：

| Episode | 使用 GT Goal 后平均全程误差 | 最优样本误差 | 平均终点误差 |
|---|---:|---:|---:|
| episode_152 | 2.18 cm | 2.16 cm | 0.19 cm |
| episode_90 | 3.02 cm | 2.97 cm | 0.37 cm |
| episode_33 | 2.70 cm | 2.61 cm | 1.18 cm |
| episode_12 | 3.50 cm | 3.36 cm | 1.85 cm |

episode_152 最典型：真实 Goal 已把终点误差降至 `0.19 cm`，但中间轨迹仍有 `2.18 cm` 平均误差。因此“Goal 不准导致整条轨迹弯曲”不是主要解释；Trajectory Decoder 确实没有恢复正确的阶段结构。

### 3.3 多次扩散采样几乎没有形成不同模式

上述 GT Goal 隔离实验中，4 条轨迹在每一帧位置上的跨样本标准差再对时间取平均，只有：

```text
episode_152: 1.28 mm
episode_90 : 1.18 mm
episode_33 : 2.18 mm
episode_12 : 1.88 mm
```

已有 `16 goals × 2 trajectories` 的训练集报告中，Best-of-K 相比 Top-1 通常只改善几毫米。蓝色杯子仿真的前五名候选也几乎相同。说明初始高斯噪声没有被映射成具有明显不同动作阶段的多模态轨迹，当前扩散器的行为更接近确定性的平滑回归器。

### 3.4 首段突变在域外场景被明显放大

当前训练集 GT 第一步通常约为亚毫米到 `1.4 mm`，训练统计的第一步 `p95` 为 `1.49 mm`。蓝色杯子仿真推理的第一步却为：

```text
local  : 8.44 mm
world  : 9.21 mm
world z: +3.07 mm
```

这不是简单的可视化误差，而是轨迹边界没有被硬钳制、时间结构弱以及仿真输入域偏移共同导致的真实首段异常。增加第 1 帧损失已将训练集上的问题压小，但它只是局部补丁，没有解决整条轨迹的阶段表示，所以平滑弧线仍然存在。

### 3.5 进度位置编码的退化是可测量的

当前实现的角度为：

```text
angle(t, i) = progress(t) × exp(-log(10000) × i / 64)
progress(t) ∈ [0,1]
```

最高频率只在整条 64 帧轨迹上走过 `1 rad`，其余频率变化更小。数值结果：

| 输入 | 相邻位置编码 L2 距离中位数 | 编码有效秩 |
|---|---:|---:|
| `progress=linspace(0,1,64)` | 0.0317 | 1.20 |
| `position=0,1,...,63` | 1.9526 | 16.90 |

因此网络虽然“有位置编码”，但它提供的时间基函数实际几乎是一维缓变信号。这个问题会直接削弱自注意力对动作阶段和折点位置的学习能力。

## 4. 为什么旧版 object_centric_diffusion 看起来更好

旧版不是因为点云或语义特征更强。相反，它每个物体只使用 64 个 XYZ 点、不使用 DINO，却在部分仿真图中更稳定地表现出“竖直抬升—水平移动—末端转向”的分段形状。主要优势来自轨迹问题的定义和时间网络，而不是输入特征复杂度。

| 对比项 | 当前 LFV Stage 2 | 旧版 object_centric_diffusion | 影响 |
|---|---|---|---|
| 去噪器 | 6 层、128 维全局 Transformer；每层一个 `k=3` 卷积 | 多尺度 Conditional 1D U-Net，`[128,256,384]`，`k=5`，Down/Up/Skip | 旧版更擅长同时建模局部动作和阶段级结构 |
| 时间位置 | `[0,1]` 弱正弦进度编码 | 卷积序列网格、下采样层级天然编码顺序和尺度 | 当前注意力很难分辨帧和阶段 |
| 起点 | 固定 Attention Token；输出再拼单位第 0 帧 | 每个去噪步骤都硬写回第 0 帧 | 旧版边界不会漂移 |
| 终点 | Goal Token + 软 endpoint loss | 每个去噪步骤都硬写回预测 Goal | 旧版严格成为“中间轨迹补全”问题 |
| 网络要生成的内容 | 生成第 1–63 帧，包括最终帧 | 只学习第 1–62 帧，首尾从 loss 中 mask | 旧版问题更明确、条件更强 |
| 场景条件 | 3 个 `128D` 全局 Token + 1 个 Goal Token | start、goal delta、absolute goal、language、16+16 个物体池化 Token，经 Transformer 得到 256D 全局条件 | 旧版进入 U-Net 前包含更明确的边界/任务信息 |
| 标签旋转 | 6D rotation，扩散中可暂时离开 SO(3) | 连续 quaternion，并显式修复符号连续性 | 当前旋转时间动力学更难，但不是平移弧线的首因 |
| 训练 | 最佳 checkpoint epoch 136，`1233` optimizer steps，batch 16，EMA 0.995 | 配置训练 1500 epochs，batch 64，SE(3) augmentation，EMA 上限 0.9999；现有仿真图用 epoch 1000 | 旧版优化和增强更充分 |
| 推理步数 | 训练评测 20；仿真 50 | 10 | 当前即使增加步数仍弧形，说明不是 DDIM 步数不足 |
| 终点显示 | 有学习误差 | 图上始终约 `0.0000 cm` | 旧版终点完美主要是硬边界构造，不应误认为全部由网络学得 |

旧版的核心形式可写成：

```text
given start and goal
    ↓
hard-clamp frame 0 and frame 63 at every DDIM step
    ↓
Conditional temporal U-Net inpaints frames 1...62
```

这比当前的：

```text
given start token and goal token
    ↓
Transformer generates frames 1...63
    ↓
soft losses encourage first/last frames to be close
```

更像一个定义良好的轨迹桥接问题。旧版不是完全没有平滑弧线；部分场景本身也是圆滑曲线。但 scene_0002、scene_0004 等结果能保留明显的先抬升后横移结构，而当前预测更常把 GT 折点圆滑化成统一弧线。

## 5. 新旧结果不能直接当作严格网络消融

还存在一个重要混杂变量：两次训练并未使用相同的数据版本。

```text
旧版根目录: /media/ljian/lj/data_3d/pouring
episode 数: 82

当前根目录: /media/ljian/lj/data_3d/pouring_lfv
episode 数: 180，其中 179 条有轨迹
```

两边有 82 个同名 episode，但 81 条两边都存在的 `dp_action_trajectory.npz` 没有一条字节一致，另有 1 条在新目录中缺失。以跟踪点初始质心作为动作锚点统计，两版数据也有不同分布：

| 指标中位数 | 旧 `pouring` | 新 `pouring_lfv` |
|---|---:|---:|
| 锚点路径总长 | 0.574 m | 0.621 m |
| 起终点直线距离 / 路径总长 | 0.535 | 0.487 |
| 距起终点直线的平均偏离 | 9.96 cm | 8.66 cm |
| 第一步位移 | 0.32 mm | 0.54 mm |
| 平均二阶差分 | 5.13 mm | 7.22 mm |

新数据并不比旧数据天然更平滑，反而路径效率更低、二阶差分更大。因此当前输出过度平滑不能仅归咎于标签已经被平滑；但由于样本集合和轨迹文件都不同，仍需要同数据、同 split 的网络消融才能定量回答“Transformer 和 U-Net 各自贡献多少”。

## 6. 数据与损失如何继续推动平滑化

### 6.1 弧长重采样抹掉了动作时序语义

预处理先从跟踪点对初始帧做 SVD，获得每一帧的 `T_0_to_t`；再定义：

```text
S[k] = Σ(||Δp|| + lambda_rot × Δtheta)
```

并在 `S` 上均匀取 64 个位置。平移采用 cubic interpolation，旋转采用 SLERP。

这样做能获得连续、固定长度的轨迹，但会产生三项副作用：

1. 停留时间和速度信息被删除；
2. 不同示范的语义阶段不会自动落在相同帧号；
3. cubic interpolation 会圆滑尖锐转折，个别情况下还可能有轻微过冲。

不过训练图中的 GT 仍然保留明显的 L 形或阶段转折，所以预处理不是当前退化的充分解释。它更像是放大“逐帧回归取条件均值”的上游因素。

### 6.2 三 Token 信息瓶颈对 Goal 和 Trajectory 的作用不同

三个 Token 的设计与此前简化 Encoder 的目标一致：

```text
initial scene
manipulated queries reference
reference queries manipulated
```

它们确实足以描述“两个物体是什么、相互关系是什么、终态应该在哪里”。这解释了 Goal 学得较好。但完整轨迹还需要至少知道：

- 在中前期先远离桌面/容器边缘的方向；
- 被操作物体在不同中间位姿下占据的空间；
- 何时完成抬升，何时开始接近参考物体；
- 末段何时开始旋转和倾倒。

当前 Max Pooling 把每个物体的所有点压成一个全局向量，双向注意力又各自 Max Pool 成一个关系向量。这些局部几何无法从三个 Token 中恢复。因此“Encoder 对 Goal 够用”不等于“Encoder 对轨迹也够用”。这不是要求重新堆叠繁杂人工特征，而是说明轨迹 Decoder 至少需要保留少量局部空间 Token 或阶段条件。

### 6.3 Joint training 使共享 Encoder 更偏向容易收敛的终态任务

当前训练直接优化：

```text
L_total = L_goal + L_trajectory
```

Goal 是单个 Pose9D，Trajectory 是 64 个带时间结构的 Pose9D。共享 Encoder 会同时接收两者梯度，但 Goal 任务更简单、信号更稳定，更容易把三 Token 表征塑造成终态关系编码。这个因素可能加剧“终态很好、中间轨迹一般”，但目前没有冻结 Encoder 或分阶段训练的消融，不能把它列为已证实的首因。

### 6.4 旋转 6D 在扩散空间中不是始终位于 SO(3)

训练和 DDIM 更新直接在 6D 连续旋转向量上进行，只有物理损失和采样结束时才投影成旋转矩阵。该设计对单个 Goal 通常足够，但对 64 帧序列会造成：

- 6D 欧氏 MSE 与 SO(3) 时间变化不完全一致；
- 当前只有平移 velocity/acceleration，没有旋转速度和旋转加速度约束；
- 旋转阶段可能提前、滞后或被平滑摊到整条轨迹。

它更可能解释姿态变化时机和旋转误差，不是平移弧线的第一原因。

## 7. 对“首段突变”的具体解释

当前 `trajectory_hard_start_token=true` 的含义容易被误读。它只保证：

1. Transformer 序列第 0 个上下文 Token 是干净单位位姿；
2. 每层 Block 后把这个 Token 重置；
3. 最终输出时删除这个 Token；
4. 采样完成后另外把单位 Pose9D 拼到输出最前面。

它**不保证**生成的第 1 帧与单位第 0 帧之间只有一个真实的小位移。第 1 帧仍然是普通扩散变量，由全局自注意力和软损失预测。因此可能出现：

```text
frame 0 = exact identity
frame 1 = already located on the learned smooth bridge
```

训练集上 `20×` 第 1 帧重构和 `2×` 起点边界损失把这个间隙压到了约 1–2.5 mm，但在仿真域外场景上又放大到 8.44 mm。由此可见，首段异常并不是轨迹采用累计位姿的计算错误，而是软边界轨迹生成在域偏移下不稳定。

## 8. 哪些直觉不应作为主要结论

1. **不是“轨迹应该改成相邻残差”就能自然解决。** 当前累计位姿语义正确，终态也因此稳定。相邻增量会把小误差累积到终点，需要单独设计积分和终点约束。
2. **不是 Goal Diffusion 的主要问题。** GT Goal 隔离实验已经证明，中间路径仍然退化。
3. **不是 DDIM 推理步数太少。** 当前仿真使用 50 步仍出现同类形状；旧版只用 10 步也能形成阶段结构。
4. **不是 `prediction_type=sample` 单独造成的。** 新旧两版都直接预测 clean sample，区别主要在时间网络和边界补全方式。
5. **不是 DINO 特征本身导致。** 旧版不使用 DINO也能获得较好的阶段路径；DINO 有利于跨实例语义，但不会自动提供轨迹阶段。
6. **不是只增加网络层数就能解决。** 如果 64 帧位置仍不可分、终点仍是软边界，更深的全局 Transformer 可能只是更有能力拟合同一条平滑平均曲线。

## 9. 建议的验证顺序

后续修改应先做单变量消融，避免再次同时改 Encoder、标签、损失和推理而无法归因。

### A. 必须先做的因果消融

1. **仅修复时间位置编码。** 保持数据、Encoder、损失和 checkpoint 训练配置不变，把帧位置改成 `0…63` 的正弦编码，或使用可学习 absolute position embedding。观察 GT 转折时间、全程误差和首步是否改善。
2. **加入严格起终点 inpainting。** 训练和每一步 DDIM 都写回单位起点与给定 Goal，loss 只覆盖中间帧。这样才与旧版问题定义对齐。
3. **在完全相同的 `pouring_lfv` 数据与 split 上，将 Transformer 替换成旧式 Conditional 1D U-Net。** 这是判断“问题是否真出在 Transformer 去噪器”的最重要 A/B 实验。
4. **固定 GT Goal 与预测 Goal 两套评测。** 每次训练都同时报告 `trajectory | GT goal` 与 `trajectory | predicted goal`，防止终态误差和路径误差再次混在一起。

### B. 若 A 仍不足，再处理信息瓶颈

5. 保留当前三个全局 Token，同时额外保留每个物体少量局部 Token，例如 `8 manipulated + 8 reference`；轨迹帧交叉注意这些局部空间 Token。不要恢复旧版繁杂人工特征。
6. 给轨迹网络显式的阶段表达，但先保持简单，例如 4 个 learned phase tokens 或多尺度时间下采样，不先引入手工 waypoint 标签。
7. 对比 Joint training、先训练 Encoder+Goal 后冻结 Goal、以及独立训练 Trajectory 三种方式，检查共享 Encoder 梯度竞争。

### C. 最后再处理标签和运动学

8. 对比弧长重采样、原始时间均匀重采样和阶段对齐重采样；检查抬升结束/倾倒开始是否落在稳定帧区间。
9. 消融 `velocity`、`acceleration` 与仿真排序中的 `30×smoothness`，判断它们是放大因素还是必要稳定项。
10. 若旋转阶段仍明显滞后，再加入 SO(3) 相邻角速度损失或在李代数增量上建模旋转；不要把它与第一轮时间编码修复混在一起。

## 10. 后续评测必须新增的指标

仅报告 64 帧平均位置误差不足以区分“轨迹整体合理”和“平滑平均曲线”。建议固定记录：

- `trajectory | GT goal` 与 `trajectory | predicted goal` 两套误差；
- 第一步位移、第一步旋转及其相对训练集 `p95` 的倍率；
- 终点硬边界误差；
- 路径总长、起终点直线距离与二者比值；
- 对起终点直线的最大偏离；
- 平移二阶差分分布，而不只报告均值；
- 抬升峰值及其发生帧、主要转折帧；
- index-aligned error 与允许速度差异的 DTW error；
- 同一 Goal 下 K 条轨迹的跨样本多样性；
- Best-of-K 改善量。若 K 增大而几乎不改善，应明确标记为 mode collapse。

## 11. 最终判断

当前问题可以明确定位到 Trajectory Diffusion 子系统，但不能简化成“Transformer 天生不适合轨迹”。更准确的说法是：**当前 Transformer 的时间位置编码发生了近退化，只有浅层单尺度局部卷积；同时轨迹任务被定义为软终点的 63 帧全生成，而不是硬起终点的中间轨迹补全；再叠加三 Token 上下文瓶颈和未对齐的逐帧回归，最终使扩散模型退化为低多样性的平滑弧线回归器。**

旧版效果相对较好的首要原因，是多尺度 Temporal U-Net 和每一步去噪的起终点硬 inpainting 给出了更强的时间归纳偏置与更明确的轨迹桥接任务；旧版终点 `0 cm` 误差则主要是构造保证，不能当作中间轨迹学习质量的证据。下一轮最有信息量的修改不是继续堆损失，而是依次完成“正确时间编码 → 起终点硬补全 → 同数据 U-Net/Transformer A/B”，再决定是否需要增加局部场景 Token 或阶段表达。

## 12. 代码与结果依据

- 当前轨迹 Decoder：`lfv/models/functional_motion_generation/trajectory/decoder.py`
- 当前轨迹损失与 DDIM：`lfv/models/functional_motion_generation/trajectory/diffuser.py`
- 当前时间编码：`lfv/models/functional_motion_generation/blocks/timestep.py`
- 当前 Transformer Block：`lfv/models/functional_motion_generation/blocks/attention.py`
- 当前三 Token Encoder：`lfv/models/functional_motion_generation/encoders/bidirectional_scene_encoder.py`
- 当前累计位姿缓存：`lfv/datasets/functional_motion/cache_builder.py`
- 当前 SVD 与弧长重采样：`lfv/pipeline/se3_trajectory.py`
- 当前仿真候选排序：`scripts/stage2/infer_sim_snapshot.py`
- 当前 checkpoint：`/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_start_fixed_v3/checkpoints/best.pt`
- 当前训练集对比图：`/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_start_fixed_v3/train_inference_visualization_ema/train_inference_gt_vs_top1_summary.png`
- 旧版 Boundary Inpainting：`/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/policy/simple_dp3.py`
- 旧版 Conditional 1D U-Net：`/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/model/diffusion/conditional_unet1d.py`
- 旧版完整配置：`/home/users1/ljian/object_centric_diffusion/config/train_dp3_goal_full64.yaml`

## 13. 当前轨迹生成网络的完整计算流程

本节记录当前代码实际执行的计算，而不是后续建议方案。当前 pouring checkpoint 的主要配置为：

| 配置 | 当前值 |
|---|---:|
| manipulated / reference 点数 | 256 / 256 |
| DINOv2 特征维度 | 384 |
| hidden dimension / attention heads | 128 / 4 |
| Goal / Trajectory Decoder 层数 | 4 / 6 |
| 轨迹长度 | 64（单位起点 + 63 个生成帧） |
| DDPM 训练时间步 | 100 |
| DDIM 默认/仿真推理步数 | 20 / 50 |
| Dropout | 0.1 |
| 总参数量 | 4,695,250 |
| Scene / Goal / Trajectory 参数量 | 646,720 / 1,207,177 / 2,841,353 |
| 单个 Trajectory Block 参数量 | 445,440 |

### 13.1 端到端结构图

~~~mermaid
flowchart LR
    A1["Manipulated XYZ<br/>B×256×3"] --> E["Three-token<br/>Scene Encoder"]
    A2["Manipulated DINO<br/>B×256×384"] --> E
    B1["Reference XYZ<br/>B×256×3"] --> E
    B2["Reference DINO<br/>B×256×384"] --> E
    E --> C["Scene context<br/>B×3×128"]
    C --> G["Goal Diffusion<br/>4-layer Transformer"]
    G --> GK["K Goal poses<br/>B×K×9"]
    C --> T["Trajectory Diffusion<br/>6-layer Transformer"]
    GK --> T
    N["M independent trajectory noises<br/>for every Goal"] --> T
    T --> O["Paired trajectories<br/>B×K×M×64×9"]
~~~

训练 Trajectory Diffusion 时，它接收 GT Goal 加小扰动；推理时接收 Goal Diffusion 生成的 K 个 Goal。每个 Goal 再从 M 份独立高斯噪声生成 M 条轨迹，因此结果保留层级对应关系：

~~~text
goal 0
 ├─ trajectory 0
 ├─ trajectory 1
 └─ ...
goal 1
 ├─ trajectory 0
 ├─ trajectory 1
 └─ ...
~~~

## 14. 从 XYZ+DINO 到三个场景 Token

### 14.1 输入与对齐关系

| 张量 | Shape | 含义 |
|---|---|---|
| <code>manipulated_points</code> | <code>[B,256,3]</code> | 杯子/被操作物体的初始可见点云 |
| <code>manipulated_dino</code> | <code>[B,256,384]</code> | 与 manipulated point 使用同一像素索引采样的 DINOv2 特征 |
| <code>reference_points</code> | <code>[B,256,3]</code> | 碗/参考物体的初始可见点云 |
| <code>reference_dino</code> | <code>[B,256,384]</code> | 与 reference point 对齐的 DINOv2 特征 |

两组点云使用 manipulated 点云质心作为共享 <code>scene_origin</code>。当前缓存 <code>scene_scale=1.0</code>。点与特征的对应关系为：

~~~text
pixel index i
 ├─ depth(pixel i) → point_xyz[i]
 └─ DINO grid(pixel i) → dino_feature[i]
~~~

Dataset 打乱点顺序时，对同一物体的 XYZ 和 DINO 使用相同 permutation。

### 14.2 当前不是先经过一个共享 PointNet

当前真实路径为：

1. manipulated 和 reference 共用一个 DINO projector；
2. XYZ 分别使用两个 role-specific projector；
3. 拼接后直接进入两个彼此独立的 PointNetBranch；
4. 两个分支输出点特征后才进行双向交叉注意力。

因此当前属于“共享 DINO 映射 + 两个独立点编码分支”，不是“先用统一 PointNet，再做两个单独编码器”。

### 14.3 单点输入编码

~~~text
shared DINO projector:
384 → LayerNorm → Linear(384,256) → GELU → Linear(256,64)

role-specific XYZ projectors:
manipulated XYZ 3 → Linear(3,64) → GELU
reference   XYZ 3 → Linear(3,64) → GELU

manipulated_input[i] = concat(manipulated_xyz_64[i], dino_64[i])
reference_input[j]   = concat(reference_xyz_64[j],   dino_64[j])
shape = [B,256,128]
~~~

DINO projector 共享，使两组语义特征位于同一学习投影空间；XYZ projector 和后续 PointNet 不共享，允许两个物体角色学习不同特征。

### 14.4 两个独立 PointNetBranch

每个分支逐点执行：

~~~text
Linear(128,128) → LayerNorm → GELU
 → Linear(128,128) → LayerNorm → GELU
 → Linear(128,128)
~~~

输出及池化：

~~~text
F_manipulated: [B,256,128]
F_reference:   [B,256,128]

g_manipulated = max over points(F_manipulated) → [B,128]
g_reference   = max over points(F_reference)   → [B,128]

concat(g_manipulated, g_reference) [B,256]
 → Linear(256,256) → GELU
 → Linear(256,128) → LayerNorm
 = initial scene token [B,128]
~~~

### 14.5 双向交叉注意力

~~~mermaid
flowchart TB
    FM["Manipulated features<br/>B×256×128"]
    FR["Reference features<br/>B×256×128"]
    FM --> MR["M queries R<br/>Q=M, K/V=R"]
    FR --> MR
    MR --> TMR["M→R token<br/>B×128"]
    FR --> RM["R queries M<br/>Q=R, K/V=M"]
    FM --> RM
    RM --> TRM["R→M token<br/>B×128"]
~~~

以 manipulated→reference 为例：

~~~text
Q = LayerNorm(F_manipulated)  [B,256,128]
K = LayerNorm(F_reference)    [B,256,128]
V = LayerNorm(F_reference)    [B,256,128]

heads = 4, head_dim = 32
attended_reference = MHA(Q,K,V) [B,256,128]
attention weights               [B,4,256,256]

concat(F_manipulated, attended_reference) [B,256,256]
 → Linear(256,256) → GELU → Dropout(0.1)
 → Linear(256,128) → LayerNorm
 → max over 256 query points
 = manipulated→reference token [B,128]
~~~

反方向完全对称。最后：

~~~text
token 0 = initial scene
token 1 = manipulated queries reference
token 2 = reference queries manipulated

scene_context =
    stack(token0, token1, token2)
    + three learned type embeddings

shape = [B,3,128]
~~~

### 14.6 信息压缩位置

~~~mermaid
flowchart LR
    P["512 point tokens<br/>512×128"] --> A["bidirectional<br/>cross-attention"]
    A --> M["three Max Pools"]
    M --> C["3 context tokens<br/>3×128"]
    C --> D["Goal and Trajectory decoders"]
~~~

轨迹网络不会再直接看到 512 个点的局部结构。交叉注意力权重可以用于解释点之间的关注关系，但没有作为稠密几何 Token 继续传给 Decoder。

## 15. Goal 如何进入轨迹网络

### 15.1 Goal 表示和 Normalizer

Goal 是轨迹第 63 帧的累计相对位姿：

~~~text
goal_pose9d = [translation_xyz, rotation_6d]
shape       = [B,9]

translation_norm = (translation - mean_train) / std_train
rotation_6d_norm = rotation_6d

translation_mean = [ 0.16675, -0.08590, -0.02227 ]
translation_std  = [ 0.12198,  0.06733,  0.08155 ]
~~~

### 15.2 Goal Diffusion

~~~mermaid
flowchart LR
    Z["K Gaussian noises<br/>B·K×9"] --> P["Pose MLP<br/>9→128→128"]
    C["Scene context<br/>B·K×3×128"] --> X["4 GoalConditionBlocks"]
    P --> X
    TS["Diffusion timestep<br/>128D"] --> X
    X --> H["LayerNorm + Linear<br/>128→9"]
    H --> D["DDIM update<br/>20 or 50 times"]
    D --> G["K Goal poses<br/>B×K×9"]
~~~

每个 GoalConditionBlock 由 goal-to-scene cross-attention 和 timestep-conditioned FFN 构成。Goal 序列只有一个 Token，因此没有 goal self-attention。网络直接预测 clean Goal。

### 15.3 训练和推理使用不同来源的 Goal

~~~text
training trajectory condition:
    normalized GT goal                         probability 0.34
    normalized GT goal + Normal(0,0.03²)       probability 0.66

inference trajectory condition:
    predicted goal → normalize → goal embedding
~~~

扰动直接加到全部 9 维，包括 translation 和 rotation 6D。训练时 Goal Diffusion 的采样结果不会输入 Trajectory Diffusion。这形成一定 train/inference condition gap，但前文 GT Goal 隔离实验已经说明它不是平滑弧线的主因。

## 16. Trajectory DDPM 训练加噪

### 16.1 干净轨迹

~~~text
trajectory_pose9d: [B,64,9]
frame 0 : identity
frame 63: goal

clean_full = normalize(trajectory_pose9d) [B,64,9]
clean      = clean_full[:,1:]             [B,63,9]
start      = clean_full[:,0]              [B,9]
~~~

删除第 0 帧是因为 Decoder 通过固定 start token 读取它，不代表生成的第 1 帧被硬约束。

### 16.2 DDPM 加噪

~~~text
t ~ Uniform{0,...,99}
epsilon ~ Normal(0,I), shape [B,63,9]

x_t = sqrt(alpha_bar_t) × clean
    + sqrt(1-alpha_bar_t) × epsilon
~~~

同一条轨迹的 63 帧共享同一个扩散时间步 t，但每帧和每个 Pose 维度的噪声独立。调度器为：

~~~text
beta_schedule   = squaredcos_cap_v2
prediction_type = sample
clip_sample     = false
~~~

所以网络直接预测干净轨迹 <code>x0</code>，而不是 epsilon。

## 17. Trajectory Decoder 的 Token 构造

### 17.1 Noisy tokens 与固定起点

~~~text
x_t[k] 9D
 → Linear(9,128) → GELU → Linear(128,128)
 = noisy_tokens [B,63,128]

normalized identity Pose9D
 → same Pose MLP
 = fixed_start_token [B,1,128]
~~~

平移 normalizer 的均值不为零，因此标准化单位起点的平移不是零向量；反标准化后仍对应准确单位位姿。

### 17.2 两种时间输入

1. diffusion timestep：当前噪声等级，整数 0–99；
2. trajectory progress：Token 在动作中的帧位置，浮点数 0–1。

~~~text
diffusion timestep 0...99
 → SinusoidalEmbedding(128)
 → Linear(128,512) → SiLU → Linear(512,128)
 → every AdaLayerNorm

trajectory progress linspace(0,1,64)
 → SinusoidalEmbedding(128)
 → directly added to pose tokens
~~~

同一模块对整数扩散时间步基本合理；编码退化发生在 trajectory progress 被压缩到 <code>[0,1]</code> 后。

### 17.3 完整序列和 Memory

~~~text
token[0]    = fixed_start_embedding + progress_embedding[0]
token[1:64] = noisy_pose_embeddings + progress_embedding[1:64]
trajectory tokens: [B,64,128]

normalized_goal 9D
 → Linear(9,128) → GELU → Linear(128,128)
 = goal token

memory = [
  initial scene token,
  manipulated→reference token,
  reference→manipulated token,
  goal token
]
shape: [B,4,128]
~~~

从这里开始，Trajectory Decoder 不再直接访问 XYZ、DINO 或点级注意力矩阵。

## 18. 单个 TrajectoryConditionBlock 的内部结构

当前共有 6 个结构相同、参数独立的 Block，每个 Block 约 445K 参数。

~~~mermaid
flowchart TB
    X0["Input trajectory tokens<br/>B×64×128"]
    T["Diffusion timestep embedding<br/>B×128"]
    M["Memory tokens<br/>B×4×128"]

    X0 --> A1["AdaLayerNorm(t)"]
    T --> A1
    A1 --> C["Conv1d<br/>128→128, kernel=3"]
    C --> R1["Residual add"]
    X0 --> R1

    R1 --> A2["AdaLayerNorm(t)"]
    T --> A2
    A2 --> SA["Full self-attention<br/>4 heads, no causal mask"]
    SA --> R2["Residual add"]
    R1 --> R2

    R2 --> A3["AdaLayerNorm(t)"]
    T --> A3
    M --> MN["LayerNorm(memory)"]
    A3 --> CA["Cross-attention<br/>Q=trajectory, K/V=memory"]
    MN --> CA
    CA --> R3["Residual add"]
    R2 --> R3

    R3 --> A4["AdaLayerNorm(t)"]
    T --> A4
    A4 --> F["FFN<br/>128→512→128"]
    F --> R4["Residual add"]
    R3 --> R4
    R4 --> O["Output tokens<br/>B×64×128"]
~~~

### 18.1 AdaLayerNorm

每个子层之前都使用扩散时间步调制：

~~~text
[shift, scale] = Linear(SiLU(timestep_embedding))

AdaLN(x,t) =
    LayerNorm_without_affine(x) × (1 + scale)
    + shift
~~~

<code>shift</code> 和 <code>scale</code> 的 Shape 都是 <code>[B,128]</code>，广播到 64 个轨迹帧。它告诉网络当前噪声等级，但不区分动作中的不同帧；帧位置只来自 progress embedding。

### 18.2 Temporal Conv1d

~~~text
AdaLN tokens [B,64,128]
 → transpose [B,128,64]
 → Conv1d(128,128,kernel=3,padding=1)
 → transpose [B,64,128]
 → residual add
~~~

一层卷积只直接读取相邻三个位置。6 层若不考虑 Attention，理论局部感受野约为 13 帧。Full Self-Attention 可以立即访问全部帧，但当前没有 U-Net 的多尺度下采样、上采样和阶段级 Skip Connection。

### 18.3 Full self-attention

~~~text
Q = K = V = AdaLN(trajectory tokens,t)
shape    = [B,64,128]
heads    = 4
head_dim = 32
mask     = none
~~~

任意一帧都可以读取另外 63 帧。整条轨迹扩散使用非因果注意力是合理的；真正的问题是 progress embedding 太弱时，注意力很难区分“Pose 内容相近但动作阶段不同”的 Token。

### 18.4 Cross-attention

~~~text
Q   = AdaLN(trajectory tokens,t) [B,64,128]
K,V = LayerNorm(memory)          [B,4,128]
output                           [B,64,128]
~~~

64 个轨迹帧交叉注意相同的四个静态 Memory Token。当前没有逐帧变化的物体几何、阶段条件、碰撞条件或当前机器人状态。

### 18.5 FFN

~~~text
AdaLN
 → Linear(128,512)
 → GELU
 → Dropout(0.1)
 → Linear(512,128)
 → residual add
~~~

### 18.6 每层后的 start token reset

每个 Block 结束后都会执行：

~~~text
tokens = concat(fixed_start_token, tokens[:,1:])
~~~

第 0 个隐藏 Token 因而不会漂移，其他帧可在下一层读取准确起点。但最终会删除该 Token：

~~~text
tokens[:,1:]
 → LayerNorm
 → Linear(128,9)
 = predicted clean trajectory [B,63,9]
~~~

这解释了为什么配置中的 <code>hard_start_token</code> 不是“扩散状态中的硬起点 inpainting”：它固定的是隐藏上下文 Token，而不是第一个生成状态。

## 19. 轨迹损失的详细组成

设 Decoder 输出为 <code>x_hat0 [B,63,9]</code>。

### 19.1 Clean-sample diffusion loss

~~~text
weight[k] = 1
weight[0] = 20
weight[62] = 2

L_diffusion =
    mean(weight × (x_hat0 - clean)²)
~~~

权重逐维广播到 9 个 Pose 分量。第一个生成帧被显著加权，目的是减小起点后的跳变。

### 19.2 物理空间位姿损失

先反标准化预测：

~~~text
pose_hat = denormalize(x_hat0)
~~~

平移损失：

~~~text
L_translation =
    SmoothL1(pose_hat[...,0:3], GT[...,0:3])
~~~

旋转测地损失：

~~~text
R_hat = rotation_6d_to_matrix(pose_hat[...,3:9])
R_gt  = rotation_6d_to_matrix(GT[...,3:9])

theta = acos(clamp((trace(R_hatᵀ R_gt)-1)/2))
L_rotation = mean(theta)
~~~

### 19.3 平移速度与加速度

预测平移前重新拼接精确零起点：

~~~text
p_hat_full = concat([0,0,0], predicted_translation)
shape      = [B,64,3]

L_velocity =
    L1(p_hat_full[k]-p_hat_full[k-1],
       p_gt[k]-p_gt[k-1])

L_acceleration =
    L1(p_hat[k+1]-2p_hat[k]+p_hat[k-1],
       p_gt[k+1]-2p_gt[k]+p_gt[k-1])
~~~

当前没有对应的 SO(3) 角速度和角加速度损失。

### 19.4 第一个生成帧边界损失

~~~text
L_start =
    MSE(first_prediction_normalized, first_GT_normalized)
  + SmoothL1(first_prediction_translation, first_GT_translation)
  + 0.5 × SO3(first_prediction_rotation, first_GT_rotation)
~~~

这里监督的是轨迹第 1 帧，不是单位第 0 帧。

### 19.5 终点损失

~~~text
L_endpoint =
    SmoothL1(last_prediction_translation, goal_translation)
  + 0.5 × SO3(last_prediction_rotation, goal_rotation)
~~~

### 19.6 当前总损失

~~~text
L_trajectory =
      1.0 × L_diffusion
    + 1.0 × L_translation
    + 0.5 × L_rotation
    + 0.5 × L_velocity
    + 0.1 × L_acceleration
    + 2.0 × L_start
    + 1.0 × L_endpoint

L_joint = L_goal + L_trajectory
~~~

Scene Encoder 同时接收 Goal 和 Trajectory 两条梯度。

## 20. DDIM 推理和 K×M 采样

### 20.1 K 个 Goal

对每个场景创建 K 个独立 9D 高斯噪声：

~~~text
goal_state ~ Normal(0,I)
shape = [B×K,9]
~~~

DDIM 反复调用 Goal Decoder，得到 <code>goals [B,K,9]</code>。

### 20.2 每个 Goal 的 M 条轨迹

~~~text
context:
[B,3,128]
 → [B,K,M,3,128]
 → [B×K×M,3,128]

goals:
[B,K,9]
 → [B,K,M,9]
 → [B×K×M,9]

trajectory_state ~ Normal(0,I)
shape = [B×K×M,63,9]
~~~

所以 M 条轨迹确实来自 M 份独立初始噪声，不是复制同一条结果。

### 20.3 每一步 DDIM

~~~mermaid
sequenceDiagram
    participant X as noisy trajectory state
    participant D as Trajectory Decoder
    participant S as DDIM Scheduler
    participant C as scene + goal memory

    loop each inference timestep
        X->>D: x_t, diffusion t, fixed start
        C->>D: 4 memory tokens
        D->>S: predicted clean trajectory x_hat0
        S->>X: previous state x_(t-1), eta=0
    end
~~~

当前循环没有执行以下边界写回：

~~~text
state[:,0]  = known start
state[:,-1] = given goal
~~~

扩散 state 只包含 frame 1–63，所以单位起点不在 state 内；最后一帧虽在 state 内，也没有在每一步 DDIM 后被写回 Goal。

### 20.4 输出整理

~~~text
denormalize translation
 → rotation_6d_to_matrix
 → matrix_to_rotation_6d
 → prepend exact identity Pose9D
 → reshape [B,K,M,64,9]
~~~

旋转的两次转换把最终输出投影回合法 SO(3)，但扩散中间状态和 Decoder 原始 6D 输出并不始终位于旋转流形上。

DDIM 使用 <code>eta=0</code>，所以给定初始噪声后反向过程是确定性的。不同初始噪声理论上仍应生成不同轨迹；当前结果相似不是因为采样代码复制了张量，而是模型把很多噪声状态映射到了相近的条件均值。

## 21. 张量 Shape 总表

以 <code>B=16</code>、<code>K=16</code>、<code>M=2</code> 为例：

| 阶段 | 通用 Shape | 示例 Shape |
|---|---|---|
| manipulated XYZ | <code>[B,256,3]</code> | <code>[16,256,3]</code> |
| manipulated DINO | <code>[B,256,384]</code> | <code>[16,256,384]</code> |
| reference XYZ | <code>[B,256,3]</code> | <code>[16,256,3]</code> |
| reference DINO | <code>[B,256,384]</code> | <code>[16,256,384]</code> |
| 每组点特征 | <code>[B,256,128]</code> | <code>[16,256,128]</code> |
| Scene context | <code>[B,3,128]</code> | <code>[16,3,128]</code> |
| 训练 clean trajectory | <code>[B,63,9]</code> | <code>[16,63,9]</code> |
| 带固定起点的隐藏序列 | <code>[B,64,128]</code> | <code>[16,64,128]</code> |
| Trajectory memory | <code>[B,4,128]</code> | <code>[16,4,128]</code> |
| Goal samples | <code>[B,K,9]</code> | <code>[16,16,9]</code> |
| 展开的轨迹 batch | <code>[B·K·M,63,9]</code> | <code>[512,63,9]</code> |
| 最终配对结果 | <code>[B,K,M,64,9]</code> | <code>[16,16,2,64,9]</code> |

## 22. 当前实现的等价伪代码

### 22.1 训练

~~~text
context = encoder(
    manipulated_xyz,
    manipulated_dino,
    reference_xyz,
    reference_dino,
)                                      # [B,3,128]

goal_loss = goal_diffusion(context, GT_goal)

trajectory_goal = normalize(GT_goal)
trajectory_goal = optionally_add_small_noise(trajectory_goal)

clean_full = normalize(GT_trajectory)   # [B,64,9]
clean      = clean_full[:,1:]           # [B,63,9]
start      = clean_full[:,0]            # [B,9]

t       = random_integer(0,99)
epsilon = random_normal_like(clean)
x_t     = DDPM.add_noise(clean, epsilon, t)

x_hat0 = trajectory_decoder(
    noisy_trajectory=x_t,
    timestep=t,
    context=context,
    normalized_goal=trajectory_goal,
    normalized_start=start,
)

trajectory_loss = all_losses(x_hat0, GT_trajectory, GT_goal)
total_loss = goal_loss + trajectory_loss
~~~

### 22.2 推理

~~~text
context = encoder(scene)

goals = goal_diffuser.sample(
    context,
    num_samples=K,
)

trajectories = trajectory_diffuser.sample(
    context,
    goals,
    num_samples_per_goal=M,
)

return goals, trajectories, goal_ids
~~~

## 23. 网络结构与当前问题的对应关系

~~~mermaid
flowchart TD
    A["512 point-level XYZ+DINO tokens"] --> B["Max Pool + relation Max Pool"]
    B --> C["only 3 global scene tokens"]
    C --> D["Goal Decoder"]
    C --> E["Trajectory Decoder"]
    D --> F["Goal learns well"]

    E1["weak progress encoding<br/>effective rank about 1.2"] --> E
    E2["single-scale temporal Conv"] --> E
    E3["soft terminal condition"] --> E
    E4["frame-wise regression on<br/>phase-misaligned labels"] --> E
    E --> G["low-frequency smooth bridge"]
    G --> H["rounded action phases"]
    G --> I["first generated frame pulled<br/>away from exact start"]
    G --> J["different noises converge<br/>to similar trajectories"]
~~~

对应的因果解释是：

1. 三个 Encoder Token 能表达两个物体的全局关系，因此 Goal Pose 容易学习；
2. Trajectory Decoder 得到 Goal 后知道终点方向；
3. 但它缺少可分辨的帧位置、局部空间 Token 和动作阶段 Token；
4. 全局 Self-Attention 容易向所有帧传播相似的 Goal-directed 信息；
5. 单尺度卷积和逐帧损失进一步把结果压成连续低频函数；
6. 最终形成“终点基本正确，但中间是统一平滑弧线”的结果。

这解释了为什么单独提高 Goal 精度、增加 DDIM 步数或继续加大第 1 帧损失不能根治问题：这些操作没有改变时间表示、边界定义和场景信息瓶颈。

## 24. 与代码逐项对应

| 文档模块 | 实现文件 |
|---|---|
| XYZ+DINO 缓存和对齐 | <code>lfv/datasets/functional_motion/cache_builder.py</code> |
| Dataset 同步 permutation | <code>lfv/datasets/functional_motion/dataset.py</code> |
| 三 Token Scene Encoder | <code>lfv/models/functional_motion_generation/encoders/bidirectional_scene_encoder.py</code> |
| 独立 PointNetBranch | <code>lfv/models/functional_motion_generation/encoders/pointnet.py</code> |
| Goal Transformer | <code>lfv/models/functional_motion_generation/goal/decoder.py</code> |
| Goal DDPM/DDIM | <code>lfv/models/functional_motion_generation/goal/diffuser.py</code> |
| Trajectory Transformer | <code>lfv/models/functional_motion_generation/trajectory/decoder.py</code> |
| Trajectory loss/DDPM/DDIM | <code>lfv/models/functional_motion_generation/trajectory/diffuser.py</code> |
| AdaLN、Attention Block | <code>lfv/models/functional_motion_generation/blocks/</code> |
| Pose9D Normalizer | <code>lfv/diffusion/normalizer.py</code> |
| Scheduler | <code>lfv/diffusion/schedulers.py</code> |
| K Goal × M Trajectory 接口 | <code>lfv/models/functional_motion_generation/system.py</code> |
