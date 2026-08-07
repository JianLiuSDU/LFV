# Stage 2 轨迹扩散逐阶段验证、修改与判断

> 日期：2026-08-07
> 任务：pouring_lfv
> 数据缓存：`/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1`
> 固定设置：train/val/test split 不变、seed=42、EMA 权重、DDIM 20 步、64 帧轨迹。

数据复核：缓存共有 179 条有效 episode，train/val/test 为 143/18/18，三者 episode ID 交集均为 0；每组 manipulated/reference 点云都固定为 256 点，DINO 维度为 384，轨迹固定为 `[64,9]`。当前源数据的 `object_instance_id` 为空，manifest 因而明确标记为 `episode_split_baseline`：本轮阶段消融内部是公平的，但这些数值不能被表述为严格的跨物体实例泛化结果。后续补齐真实实例 ID 后必须重建 split 再做最终泛化报告。

## 1. 为什么采用逐阶段消融

本轮不把所有结构修改一次性叠加，而是使用下列顺序：

| 阶段 | 唯一新增因素 | 目的 | 状态 |
|---|---|---|---|
| A0 | 旧模型，归一化 0–1 时间编码 | 建立可复现基线 | 已完成 |
| A1 | 离散帧位置编码 0–63 | 验证原时间编码退化的影响 | 已完成 |
| A2a | 直接替换式 Goal-conditioned Context Mixer | 检验先交互再解码 | 已完成，负向 |
| A2b | 小尺度门控式 Context Mixer | 保留 A1 原始 memory，仅学习条件残差 | 已完成，保留 |
| A3a | 4 个直接注入的 latent phase tokens | 显式表示粗阶段 | 已完成，负向 |
| A3b | phase-only gated residual | 检验阶段 Token 还是注入尺度导致退化 | 已完成，保留 |
| A4 | 局部/全局时间注意力交替和门控残差 | 同时保留局部动态与全局连贯性 | 已完成，负向 |

每一阶段均重新从随机初始化训练。旧权重不能直接更换时间编码后作为 A1，因为那会改变旧权重所处的基函数而无法解释结果。

## 2. 固定诊断协议

除原有终态与轨迹误差外，本轮新增 endpoint-detrended temporal spectrum 审计：

1. 对每条平移轨迹减去其起点到终点的直线桥接；
2. 对剩余路径形状以及逐帧速度分别做正交 DCT-II；
3. 频带固定为 low=`[1,5)`、mid=`[5,17)`、high=`[17,33)`；
4. 报告各频带能量保留率和预测/GT 系数余弦；
5. 额外报告首步位移、路径长度比、去趋势形状相对 L2，以及主曲率峰所在帧的误差；
6. 分别在 GT goal 和 predicted goal 条件下运行，从而分离 trajectory decoder 自身问题与 goal error propagation。

频率能量保留率接近 1 仅表示幅值接近；系数余弦接近 1 才表示相位和方向一致。因此不能通过单独增强高频能量来判断改进是否成功。

## 3. A0：旧轨迹模型基线

### 3.1 模型与检查点

- checkpoint：`full_joint_start_fixed_v3/checkpoints/best.pt`
- best epoch：136，global step：1233
- 结构：hidden=128，6 个 trajectory blocks，4 heads
- 时间位置编码：`legacy_normalized_sinusoidal`，即 64 帧仅覆盖 0–1
- 测试集：18 episodes

### 3.2 GT goal 条件结果

| 指标 | A0 数值 |
|---|---:|
| endpoint translation error | 0.01058 m |
| first-step prediction / GT | 0.002524 / 0.001069 m |
| path length ratio | 0.7318 |
| detrended shape relative L2 | 0.4569 |
| dominant curvature frame error | 15.44 帧 |
| low position energy retention / cosine | 0.8890 / 0.8753 |
| mid position energy retention / cosine | 0.2602 / 0.3771 |
| high position energy retention / cosine | 0.3163 / 0.0957 |
| mid velocity energy retention / cosine | 0.1728 / 0.0672 |

结果目录：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a0_legacy/spectrum_gt_goal`。

### 3.3 Predicted goal 条件结果

| 指标 | A0 数值 |
|---|---:|
| endpoint translation error | 0.03268 m |
| first-step prediction / GT | 0.002472 / 0.001069 m |
| path length ratio | 0.7439 |
| detrended shape relative L2 | 0.4724 |
| dominant curvature frame error | 15.56 帧 |
| low position energy retention / cosine | 0.9111 / 0.8757 |
| mid position energy retention / cosine | 0.2606 / 0.3565 |
| mid velocity energy retention / cosine | 0.1819 / 0.0536 |

结果目录：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a0_legacy/spectrum_predicted_goal`。

### 3.4 A0 判断

旧模型不是“所有频率都不足”：低频形状已经学习较好，而中频位置只保留约 26%，中频速度只保留约 17%，且相位一致性很差。主转折平均错约 15.5 帧，路径长度被压缩到 GT 的约 73%。另一方面，首步位移却是 GT 的约 2.36 倍，这是一个不希望出现的局部高频跳变。因此正确方向不是无差别增强高频，而是恢复任务真实的中频阶段结构，同时抑制首帧伪高频。

GT goal 与 predicted goal 的形状频谱结果高度接近，仅终点误差明显受 Goal 预测影响。这证明当前弧线化和错转折的主要责任在 Trajectory Diffusion Transformer，而不是 Goal Decoder。

## 4. A1：离散帧位置编码

训练配置：`configs/stage2/ablation_a1_discrete_position.yaml`。除把时间编码改为离散帧索引 0–63 以及独立输出目录外，其余超参数与 A0 的训练配置一致。

### 4.1 训练结果

- early stop：epoch 233；best checkpoint：epoch 153；
- best validation total：0.44689；
- 标准 16 goals × 2 trajectories 测试：trajectory top-1 translation 从 A0 的 0.04494 m 降到 0.04075 m，约改善 9.3%；
- goal top-1 translation 为 0.02700 m，未因轨迹位置编码修改而退化。

### 4.2 GT goal 频谱结果

| 指标 | A0 | A1 | 变化 |
|---|---:|---:|---:|
| endpoint translation error / m | 0.01058 | 0.00920 | -13.0% |
| first-step prediction / m | 0.002524 | 0.002376 | -5.9% |
| path length ratio | 0.7318 | 0.7851 | 更接近 1 |
| detrended shape relative L2 | 0.4569 | 0.4460 | -2.4% |
| dominant curvature frame error | 15.44 | 8.67 | -43.9% |
| low position retention | 0.8890 | 0.9450 | +0.0560 |
| mid position retention | 0.2602 | 0.3570 | +0.0968 |
| mid position cosine | 0.3771 | 0.3660 | -0.0111 |
| mid velocity retention | 0.1728 | 0.3272 | +0.1544 |
| mid velocity cosine | 0.0672 | 0.3149 | +0.2477 |

### 4.3 Predicted goal 频谱结果

Predicted-goal 条件下，主转折帧误差从 15.56 降到 9.39，中频位置能量从 0.2606 升到 0.3437，中频速度系数余弦从 0.0536 升到 0.3284。变化方向与 GT-goal 一致。

### 4.4 A1 判断

离散时间位置编码是有效且必要的修复。最明确的收益不是“提高所有高频”，而是：轨迹的主转折时间明显更准确，中频速度的幅值与相位均得到恢复，路径长度不再被压缩得那么严重。它没有完全解决问题：中频位置能量仍只有 GT 的约 35.7%，系数余弦仅 0.366；首步仍约为 GT 的 2.22 倍。因此 A2 应继续验证终态是否需要先与场景关系 Token 交互，而不能把静态 Goal Token 直接并入 memory。

结果目录：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a1_discrete_position`。

## 5. A2：Goal-conditioned Context Mixer

A2a 在 A1 上只增加两层 pre-norm Transformer Encoder：输入为三个 Scene Tokens 与一个 Goal Token，先形成四个 goal-conditioned memory tokens，再供原有 6 个轨迹块 cross-attention。它不增加 phase token、不改变时间注意力范围，也不启用残差门控。

### 5.1 A2a 训练与测试

- early stop：epoch 131；best checkpoint：epoch 51；best validation total：0.48928；
- EMA top-1 trajectory translation：0.05683 m，明显差于 A1 的 0.04075 m；
- raw top-1 trajectory translation：0.04758 m，说明早期 best checkpoint 的 EMA 滞后解释了一部分退化，但 raw 仍差于 A1；
- EMA predicted first step：0.00436 m，A1 为 0.00228 m；
- GT-goal endpoint error：0.03927 m，A1 为 0.00920 m。

### 5.2 A2a 频谱判断

| GT-goal 指标 | A1 | A2a |
|---|---:|---:|
| low position retention / cosine | 0.9450 / 0.8937 | 0.5074 / 0.8258 |
| mid position retention / cosine | 0.3570 / 0.3660 | 0.2121 / 0.2724 |
| high position retention / cosine | 0.2511 / 0.0548 | 0.8713 / 0.0580 |
| mid velocity retention / cosine | 0.3272 / 0.3149 | 0.2572 / 0.1423 |
| dominant curvature frame error | 8.67 | 9.94 |

A2a 是一个重要反例：高频能量看起来从 25.1% 升到 87.1%，但高频系数余弦仍只有 0.058，而且首步跳变扩大。这不是学会了真实高频，而是引入了无正确相位的伪高频。直接用 mixer 输出替换原始 `[scene, goal]` memory，使已有效的低频和终点条件路径受到破坏。

因此不把 A2a 原样堆叠进下一阶段。代码保留 A2a 以便复现，同时增加 A2b：

```text
M0 = concat(scene_tokens, goal_token)
Mmix = TransformerMixer(M0)
M = M0 + gamma_ctx * (Mmix - M0)
gamma_ctx initial = 0.1, learnable per channel
```

这使 A2b 初始化时接近已验证的 A1，mixer 只能以小尺度残差逐渐提供 goal-conditioned 修正。A2b 仍不启用 phase token 或局部注意力，因而可以单独判断门控式上下文交互是否有效。

### 5.3 A2b 结果与判断

- early stop：epoch 204；best checkpoint：epoch 124；best validation total：0.43626；
- trajectory top-1 translation：0.04107 m，与 A1 的 0.04075 m 基本持平（+0.32 mm）；
- trajectory rotation：14.69°，优于 A1 的 15.01°；
- predicted first step：0.00204 m，优于 A1 的 0.00228 m；
- goal top-1 translation/rotation：0.02547 m / 25.03°，均优于 A1；
- GT-goal mid-position retention/cosine：0.3793/0.3818，优于 A1 的 0.3570/0.3660；
- predicted-goal 主转折帧误差：8.06，优于 A1 的 9.39。

A2b 没有显著降低逐帧平移误差，但消除了 A2a 的伪高频退化，并在旋转、首步边界、中频形状和 predicted-goal 转折时间上取得一致的小幅收益。因此结论不是“Context Mixer 越深越好”，而是：goal-scene 预交互必须保留原始 memory 的短路径，并以小尺度残差进入。A2b 可以作为 A3 的基础，但其收益量级较小，不能替代显式阶段建模。

结果目录：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a2b_gated_goal_context`。

## 6. A3：有序 Latent Phase Tokens

A3 在 A2b 上增加 4 个 learned phase queries。它们先 cross-attend 门控后的 scene-goal memory，再做一次 phase self-attention；每个轨迹 Block 新增 trajectory-to-phase cross-attention。帧到阶段的 attention logit 叠加宽度 `sigma=0.22` 的高斯单调软偏置，但没有人工阶段标签，也不强制硬切分帧段。A3 暂不改变原来的全局 temporal self-attention，也不启用 Block 分支 LayerScale，从而单独判断“粗阶段潜变量”是否有效。

### 6.1 A3a 结果

- early stop：epoch 196；best checkpoint：epoch 116；best validation total：0.43385；
- trajectory top-1 translation/rotation：0.04395 m / 14.33°；
- predicted first step：0.00171 m，是目前最接近 GT 0.00126 m 的阶段；
- GT-goal mid-position retention/cosine：0.2576/0.3617，明显低于 A2b 的 0.3793/0.3818；
- GT-goal dominant curvature frame error：9.33 帧，差于 A2b 的 8.56；
- predicted-goal mid-velocity retention/cosine：0.2103/0.2730，也低于 A2b。

A3a 表明 phase tokens 对边界和旋转有帮助，但以单位尺度在 6 个 Block 中反复注入，会压制已有的中频路径结构。这个结果不能简单解释为“阶段 Token 无效”，因为新增分支的尺度与原来已经训练稳定的四个残差分支相同。

因此增加 A3b，只对新增 phase update 使用门控：

```text
x <- x + gamma_phase * PhaseCrossAttention(x, phase_tokens)
gamma_phase initial = 0.1, learnable per channel
```

原有 temporal conv、temporal self-attention、scene-goal cross-attention 与 FFN 仍保持普通残差；时间注意力仍为全局。这样能单独判断 A3a 的退化究竟来自阶段概念，还是来自不受控的新增残差幅值。

### 6.2 A3b 结果与判断

- early stop：epoch 221；best checkpoint：epoch 141；best validation total：0.44083；
- trajectory top-1 translation/rotation：0.04290 m / 14.00°；
- predicted first step：0.00184 m；
- GT-goal dominant curvature frame error：6.22 帧，是此前最好的 A2b 8.56 帧的进一步明显改善；
- predicted-goal dominant curvature frame error：7.67 帧；
- GT-goal mid-position retention/cosine：0.3243/0.3786；
- GT-goal mid-velocity retention/cosine：0.2747/0.3232。

门控后，A3a 的主要退化被部分修复：相比 A3a，GT-goal 转折误差从 9.33 降到 6.22，首步仍明显优于 A2b，旋转误差也继续下降。这证明阶段 Token 的核心价值是帮助网络定位“什么时候转折”。但它没有恢复足够的中频幅值：mid-position retention 仍低于 A2b 的 0.3793，平均平移也未超过 A1/A2b。下一阶段应针对局部时间建模，而不是进一步增加阶段 Token 数量或提高 phase 注入强度。

结果目录：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a3b_gated_phase_tokens`。

## 7. A4：局部/全局时间注意力交替与 Block 门控残差

A4 保留 A3b 的门控 context 与 4 个 phase tokens，将 6 个轨迹 Block 的 temporal self-attention 改为：第 0/2/4 层仅看 7 帧局部窗口，第 1/3/5 层保持全局注意力。同时为每个 Block 的 temporal conv、self-attention、phase cross-attention、scene-goal cross-attention 与 FFN 残差增加逐通道 LayerScale，均从 0.1 开始学习。目标是让局部层恢复真实中频动态、全局层保持终点与整段一致性，并避免任何新增分支在训练初期压过原表示。

### 7.1 A4 训练与测试结果

- early stop：epoch 200；best checkpoint：epoch 120；best validation total：0.43957；
- trajectory top-1 translation/rotation：0.04444 m / 14.71°；
- endpoint translation/rotation：0.02969 m / 25.57°；
- predicted first step：0.00178 m，仍接近 GT 的 0.00126 m；
- GT-goal endpoint translation error：0.01471 m；
- GT-goal dominant curvature frame error：8.11 帧；
- GT-goal mid-position retention/cosine：0.3139/0.3372；
- GT-goal mid-velocity retention/cosine：0.2311/0.2986；
- predicted-goal mid-position retention/cosine：0.3018/0.3368。

这些数值均低于 A3b 在旋转、转折时刻和中频相位上的结果，也低于 A1/A2b 在平均平移和中频幅值上的结果。A4 的首步边界仍然良好，但这主要来自 A3b 已有的 phase gate，而不是局部/全局注意力本身的新收益。

### 7.2 门控参数检查

A4 最佳 EMA 权重中，Context Mixer 的逐通道门控均值为 0.0958；6 个 Block 的 `conv/self/phase/cross/ffn` LayerScale 均值都保持在约 0.0997–0.1027，几乎停留在初始化值 0.1 附近。它们没有塌缩到 0，也没有被学习成明显不同的分支尺度。这说明：

1. 门控成功避免了不受控的大残差，但当前监督没有促使网络形成清晰的分支分工；
2. 把所有既有分支同时压到 0.1，会降低整条主干的有效更新幅度；
3. 7 帧局部窗口并没有自动恢复中频阶段形状，反而削弱了全局路径建模。

### 7.3 A4 判断

A4 是负向消融，不设为默认配置。问题不在于“Transformer 缺少残差连接”：原始 Block 本身已经对 temporal conv、self-attention、cross-attention 和 FFN 使用标准残差。A4 新增的是可学习 LayerScale，而不是补上此前不存在的残差。实验说明，在当前 6 层、128 维的小模型和 143 条训练轨迹上，给所有分支统一加小门控及硬编码的局部/全局交替并不合适。

代码会保留 `temporal_attention_mode`、`local_attention_window` 和 `residual_gating` 配置开关，以便未来在更大数据上重测；当前推荐配置仍采用全局 temporal self-attention，仅对新增的 context/phase 分支做小尺度门控。

结果目录：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a4_local_global_gated`。

## 8. A0–A4 统一结果汇总

下表全部来自同一个 test split、EMA、seed=42、16 goals × 2 trajectories 的评估；频谱指标使用 GT goal，从而尽量隔离 Goal Decoder 的误差传播。

| 阶段 | 轨迹平移 / m ↓ | 轨迹旋转 / ° ↓ | 首步预测 / m | 中频位置 retention ↑ | 中频位置 cosine ↑ | 转折帧误差 ↓ | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | 0.04494 | 14.72 | 0.00252 | 0.2602 | 0.3771 | 15.44 | 旧基线 |
| A1 | **0.04075** | 15.01 | 0.00238 | 0.3570 | 0.3660 | 8.67 | 必要修复 |
| A2a | 0.05683 | 15.25 | 0.00489 | 0.2121 | 0.2724 | 9.94 | 直接替换式交互失败 |
| A2b | 0.04107 | 14.69 | 0.00232 | **0.3793** | **0.3818** | 8.56 | 保留，平衡型 |
| A3a | 0.04395 | 14.33 | 0.00214 | 0.2576 | 0.3617 | 9.33 | 无门控阶段注入失败 |
| A3b | 0.04290 | **14.00** | **0.00198** | 0.3243 | 0.3786 | **6.22** | 保留，阶段时序型 |
| A4 | 0.04444 | 14.71 | 0.00200 | 0.3139 | 0.3372 | 8.11 | 局部/全局+全门控失败 |

注：表中的“首步预测”来自固定 GT-goal 频谱采样，GT 为 0.001069 m；标准多样本评估中各阶段首步数值略有不同，这是采样协议不同造成的，阶段间排序基本一致。

完整对比图：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/comparison/ablation_comparison.png`；机器可读汇总：同目录 `ablation_summary.json`。

### 8.1 推荐模型不是单一指标冠军

- 若优先追求平均位置误差，使用 **A1**；它证明离散帧位置编码是收益最大、风险最低的修复。
- 若希望位置、中频结构和终态条件交互更均衡，使用 **A2b**；它的中频幅值与相位最好。
- 若执行任务更关心转折时刻、首步稳定性和旋转，使用 **A3b**；它的转折帧误差最低，旋转最好。
- 不推荐 A2a、A3a 或 A4 作为当前默认值。

目前没有一个结构同时赢得所有指标，不能因为 A3b 的阶段时刻更好就宣称轨迹问题已彻底解决。推荐把 A3b 作为下一轮结构研究起点，同时保留 A1/A2b 作为强基线。

## 9. 两个扩散分支共享 Encoder 的梯度诊断

对固定 train split、seed=42 的 8 个 batch，分别只对 Goal loss 和 Trajectory loss 反向传播到共享 Scene Encoder，统计梯度范数和余弦：

| 阶段 | Goal 梯度范数 | Trajectory 梯度范数 | Traj/Goal | 平均余弦 | 负余弦 batch 比例 |
|---|---:|---:|---:|---:|---:|
| A0 | 3.199 | 1.220 | 0.388 | 0.0338 | 25.0% |
| A1 | 2.932 | 1.205 | 0.425 | 0.0197 | 62.5% |
| A2b | 3.294 | 1.132 | 0.362 | 0.0279 | 50.0% |
| A3b | 3.035 | 1.110 | 0.376 | 0.0085 | 37.5% |
| A4 | 3.459 | 0.878 | 0.258 | 0.0357 | 37.5% |

判断如下：

1. 平均余弦接近 0，而不是持续显著小于 0，因此没有证据支持“两个任务必然互相破坏、必须拆 Encoder”；
2. 单 batch 确实会出现负余弦，且不同结构下比例为 25%–62.5%，说明小数据训练噪声和局部梯度冲突真实存在；
3. 更稳定的问题是尺度不平衡：Trajectory 对 Encoder 的梯度只有 Goal 的约 0.26–0.43 倍，共享表示更容易被 Goal 任务主导；
4. A4 的 Trajectory/Goal 比最低，也与其轨迹收益不足一致，但这里只能认为相关，不能据此断言因果。

因此联合训练公式本身是合理的：

```text
L_total = lambda_goal * L_goal + lambda_traj * L_trajectory
grad_encoder = lambda_goal * grad(L_goal) + lambda_traj * grad(L_trajectory)
```

下一轮优先验证的不是拆分 Encoder，而是仅对共享参数使用动态梯度平衡（如 GradNorm）或先扫描 `lambda_traj/lambda_goal`。若扩大 batch 后仍长期出现负余弦，再考虑 PCGrad；不能直接根据当前 8-batch 诊断引入复杂多任务优化器。

每个阶段的原始报告位于对应实验目录下的 `encoder_gradient_report.json`。

## 10. “轨迹太弧形”是否等于缺少高频

结论是：**部分相关，但更准确的问题是中频阶段结构欠拟合、相位/转折时间不准，同时存在首帧伪高频；并不是所有高频都越多越好。**

证据链：

1. A0 的低频 retention/cosine 已达到 0.889/0.875，但中频位置 retention 只有 0.260、中频速度 cosine 只有 0.067，且转折错 15.44 帧；这符合“只学到平滑大弧线”的现象。
2. A1 只修正位置编码，中频速度 retention/cosine 就从 0.173/0.067 提升到 0.327/0.315，转折误差降至 8.67 帧，说明网络容量并非完全不能表达动态，原时间基函数本身就是重要瓶颈。
3. A2a 的高频位置能量 retention 达到 0.871，却因高频 cosine 仅 0.058、首步跳变巨大而整体最差。这直接否定了“增加高频能量即可”的方案。
4. A3b 的中频能量不是最高，但通过 phase tokens 把转折误差降到 6.22 帧，说明阶段时序条件比盲目放大频率更关键。
5. A4 的局部注意力没有继续改善，说明把感受野限制在 7 帧不等于自动学到局部细节。

因此下一轮应按以下优先级推进：

1. **先做共享 Encoder 的 loss/gradient balance 扫描**，避免轨迹监督在共同表征中被 Goal 分支压小；
2. 以 A3b 为起点，给轨迹 denoiser 增加一个轻量、多尺度且保持全局旁路的 temporal branch，例如 dilation=`1,2,4` 的深度可分离 1D conv，再以零/小门控残差注入；
3. 对真实轨迹残差或速度使用多分辨率频谱损失，但必须同时约束复数/带符号 DCT 系数或时域导数，不能只匹配能量；
4. 保留首步边界损失和 phase tokens，继续监测 first-step ratio、mid-band cosine 和 curvature-frame error；
5. 每项都从 A3b 单独消融，不把梯度平衡、多尺度卷积和频谱损失一次性叠加。

这里建议的多尺度卷积并非退回旧 `object_centric_diffusion` 特征工程，而是在当前 Diffusion Transformer 前/块内提供短、中、长时间尺度的可学习残差路径；scene encoder、goal diffusion、trajectory diffusion 的总体框架不变。

## 11. 本轮实际代码修改

### 11.1 诊断基础设施

- `lfv/evaluation/functional_motion/spectrum.py`：endpoint detrend、DCT 频带和阶段时序指标；
- `scripts/stage2/analyze_trajectory_spectrum.py`：对任意旧/新 checkpoint 运行 GT-goal 或 predicted-goal 诊断；
- `scripts/stage2/analyze_shared_encoder_gradients.py`：分离两个 loss 对共享 Encoder 的梯度；
- `scripts/stage2/plot_trajectory_ablation.py`：汇总所有阶段并输出固定对比图；
- `tests/stage2/test_trajectory_spectrum.py`：验证去趋势、频带与指标 shape/数值行为。

### 11.2 网络结构开关

- `lfv/models/functional_motion_generation/blocks/conditioning.py`：门控 Goal-conditioned Context Mixer 与有序 Latent Phase Token Generator；
- `lfv/models/functional_motion_generation/blocks/attention.py`：phase cross-attention、单调高斯 phase bias、局部/全局 temporal attention 和可选 LayerScale；
- `lfv/models/functional_motion_generation/trajectory/decoder.py`：统一组装 A1–A4，并保持旧 checkpoint 默认配置兼容；
- `lfv/models/functional_motion_generation/system.py` 与 `loading.py`：训练构建、保存配置和旧权重恢复；
- `tests/stage2/test_stage_aware_trajectory_decoder.py`：新结构 forward、mask、门控与排列的单元测试。

### 11.3 可复现实验配置

`configs/stage2/ablation_a1_discrete_position.yaml` 到 `ablation_a4_local_global_gated.yaml` 固定了每个阶段唯一变化。所有阶段均从随机初始化训练，输出完整 best/last checkpoint、训练历史、标准测试、GT/predicted-goal 频谱和必要的梯度报告。

最终回归结果：`tests/stage2` 共 **24 passed**。

## 12. 固定样本可视化与最终结论

选择 A3b EMA，在固定训练样本 `episode_12/33/90/152` 上生成 8 个 Goal、每个 Goal 2 条轨迹，并将 top-1 与 GT 的 64 帧坐标系画回原图。输出为：

- 轨迹总图：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a3b_gated_phase_tokens/train_inference_visualization_ema/train_inference_gt_vs_top1_summary.png`；
- 终态总图：同目录 `train_goal_pose_gt_vs_top1_summary.png`；
- 每个 episode 的独立图与数值：同目录下各 PNG 和 `train_inference_report.json`。

这些训练样本中，终态平移误差约 10–18 mm、旋转误差约 0.44–2.24°，再次印证 Goal Decoder 学得明显好于完整中间轨迹；轨迹平均平移误差约 17–24 mm，主要差异仍集中在路径形状而非最后一帧。

本轮最终判断是：

1. 离散帧位置编码必须合入默认实现；
2. Transformer 本来已有标准残差，新增条件分支必须使用小尺度残差门控；
3. phase tokens 有效改善阶段转折，但尚未解决中频幅值不足；
4. 对所有 Block 做统一 LayerScale 和局部/全局硬交替无收益；
5. 共享 Encoder 可以继续使用，但需要解决 Goal 梯度占优；
6. 下一轮应在 A3b 上依次验证梯度平衡、多尺度时域残差和带相位约束的频谱监督，而不是一次性堆叠。

## 13. A3b 蓝色杯子仿真推理

### 13.1 固定输入与推理设置

- 仿真：ManiSkill pouring，`Cole_Hardware_Mug_Classic_Blue` 杯子与红色碗；
- 快照：`blue_mug_start_fixed_seed_0/snapshot/pouring_snapshot.npz`；
- 模型：A3b best checkpoint epoch 141，EMA 权重；
- 输入：杯子与碗分别采样 256 点，每个点对应 384 维 DINOv2 特征；XYZ 和 DINO 严格使用相同像素索引；
- 采样：16 个 Goal，每个 Goal 2 条轨迹，Goal/Trajectory 均使用 50 步推理；
- 排序：只使用训练集 Goal 先验、轨迹到 Goal 的终点一致性、二阶差分、最大步长和首步异常量，不使用仿真 GT 终态挑选候选。

### 13.2 选中结果

| 指标 | A3b 仿真结果 |
|---|---:|
| selected goal / trajectory | 15 / 0 |
| predicted relative rotation | 107.63° |
| trajectory endpoint to sampled goal | 4.82 mm / 2.45° |
| mean second difference | 2.10 mm |
| first step, local | 5.26 mm / 0.83° |
| first step, world | 4.40 mm |
| maximum step | 18.12 mm |
| training first-step P95 | 1.49 mm |

可视化表明模型从蓝色杯子当前位姿先抬升，再向碗上方移动并逐渐旋转；预测 Goal 位于碗的上方空间而不是碗表面，这是 pouring 任务所需的倾倒工作位姿。双向 Encoder 注意力分别落在杯子与碗的可见区域，输入角色没有交换。

但仿真首步 5.26 mm 仍是训练分布 P95 的约 3.5 倍，因此不能仅凭轨迹看起来连续就直接认定可以安全执行。相比旧 start-fixed 模型，本次轨迹到 Goal 的终点平移差约从 17.2 mm 降至 4.8 mm，最大单步约从 28.1 mm 降至 18.1 mm，已有改善；首步跨域跳变仍需在下一轮通过 loss/gradient balance 和仿真输入域适配继续处理。

### 13.3 固定可视化输出

- 一张式总览：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a3b_gated_phase_tokens/sim_blue_mug_seed_0/simulation_inference_summary.png`；
- 64 帧坐标系：同目录 `full64_coordinate_frames_overlay.png`；
- Goal 候选：同目录 `goal_pose_candidates_overlay.png`；
- Encoder 双向注意力：同目录 `encoder_cross_attention_summary.png`；
- 完整采样张量：同目录 `functional_motion_prediction.npz`；
- 数值与候选排序：同目录 `motion_inference_report.json`。

`scripts/stage2/infer_sim_snapshot.py` 已固定自动生成上述单图、分图、NPZ 和 JSON。后续更换模型 checkpoint 或仿真快照时不需要重新编写可视化代码。

## 14. 复用旧 GraspNet 位姿的仿真抓取与轨迹执行

### 14.1 执行设置

- 抓取：复用 `episode_0_to_cole_blue_mug_seed_0/topdown_grasp/graspnet_selected_object.npy`；
- 机器人：`panda_long_finger`，延长指面接触面积约为原始 Panda 指面的 6.49 倍；
- 抓取局部偏移：`[0.005,-0.005,0] m`；预抓取距离 0.08 m；
- 到达抓取位姿后显式发送归一化 `-1.0` 完全闭合指令，共 35 步，再保持 10 步；
- 后续运动：执行 A3b 在相同仿真快照上选中的 Full64 物体轨迹，每个模型帧插值 3 个控制子步；
- 录像：斜视角 1280×720 与正前方 640×480，同时以 30 FPS 记录。

### 14.2 抓取和跟踪结果

| 指标 | 结果 |
|---|---:|
| snapshot/execution 初始对齐误差 | 0.00000003 m |
| 完全闭合前双指关节 | 0.0400 / 0.0400 m |
| 完全闭合后双指关节 | 0.01267 / 0.01265 m |
| 闭合后抓取检测 | true |
| 轨迹结束抓取检测 | true |
| 中途丢失抓取帧 | 无 |
| TCP 平均跟踪误差 | 4.81 mm |
| TCP 最大跟踪误差 | 14.67 mm |
| 物体最终位置跟踪误差 | 20.71 mm |

这证明第一阶段保存的 top-down GraspNet 位姿、完全闭合控制和延长指面能够稳定抓住杯子，并在 A3b 的整段抬升/旋转轨迹中保持抓取。当前失败不是夹爪再次松脱。

### 14.3 任务失败判断

环境 `simulator_success=false`。实际最终杯子世界位置为 `[-0.0881,0.0319,0.1657] m`，碗中心初始位置为 `[0.0600,0.1800,0.0357] m`，二者平面距离约 20.9 cm；预测终点与碗中心的平面距离本身也约 20.5 cm。关键帧显示杯子被可靠抬起并倾斜，但停在碗的左上方，没有进入碗上方的有效 pouring 区域。

因此本次链路应拆成两个结论：

1. **抓取与执行控制成功**：旧抓取位姿可复用，夹爪闭合、抓取保持和 TCP 跟踪均正常；
2. **Stage 2 跨域 Goal 失败**：仿真 Goal 的 XY 关系预测错误，轨迹只是较好地跟随了一个错误终态，不能计为完整 pouring 成功。

### 14.4 录像与结果

- 斜视角录像：`/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a3b_gated_phase_tokens/sim_blue_mug_seed_0/execution_previous_grasp/pouring_execution.mp4`；
- 正前方录像：同目录 `pouring_execution_front.mp4`；
- 执行报告：同目录 `execution_report.json`；
- 抓取闭合与第 16/32/48/64 帧关键帧：同目录 `keyframe_*.png`。

两段录像均已通过 `ffprobe` 校验，共 348 帧、30 FPS、11.6 秒。该执行目录不覆盖此前成功/失败录像，可作为后续 Goal 域适配前的固定 A3b 仿真基线。
