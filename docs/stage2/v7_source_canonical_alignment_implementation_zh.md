# Stage 2 V7：Source-Canonical Functional Alignment 实现说明

本文是 V7 的代码与实验交付说明，不替换 V2/V6 的论文材料。V7 的注册名为
`v7_functional_alignment`，推荐配置中的模式为
`motion_field_mode: local_functional_bottleneck`。旧模型仍由
`three_token_hierarchical_diffusion` 注册名加载，旧 checkpoint 不需要转换。

## 1. 设计目标和实际代码映射

V7 将运动相关性限制为一个真正的逐点标量场。Selector 可以读取两个物体的关系，
但 Selector 的隐状态不能作为生成器条件；它只能产生 manipulated/reference 两个
标量 Field。生成器重新从原始逐点 XYZ--DINO payload 建立关系，并在 key attention、
query 输出和 pooling 三处使用 Field。因此把 Field 置零会使 generator context 置零，
而不是仍然通过一个全局特征旁路完成预测。

| 功能 | 实际文件/类 | V2/V6 兼容说明 |
|---|---|---|
| V2 Joint、V6 Balanced Long 主模型 | `lfv/models/functional_motion_generation/system.py` 的 `ThreeTokenHierarchicalDiffusion` | 文件和 registry 名称保持不变；V2/V6 仅由配置项区分 |
| V2/V6 XYZ+DINO 编码器、双向 cross-attention、joint/marginal/three-token field | `encoders/bidirectional_scene_encoder.py` 的 `BidirectionalSceneEncoder` | 只供旧 registry 使用 |
| V7 LocalPointEncoder、FieldSelector、GatedRelationEncoder、FunctionalPooling | `encoders/v7.py` | V7 独立实现，无 PointNet global pooling |
| V7 系统模型 | `v7_system.py` 的 `V7FunctionalAlignmentDiffusion` | 复用现有 Goal/Trajectory diffuser |
| Goal Pose Diffusion | `goal/decoder.py`, `goal/diffuser.py` | V7 只改变 context 序列长度，不改公开 API |
| Trajectory Diffusion | `trajectory/decoder.py`, `trajectory/diffuser.py` | V7 复用现有 64×9D 轨迹主干 |
| Stage 2 模型 registry/加载 | `registry.py`, `loading.py` | V7 参数条件化注入；旧参数不会传入旧类 |
| 训练、EMA、AMP、梯度裁剪、断点 | `lfv/training/functional_motion/trainer.py` | V7 额外调用 `set_training_progress` 实施 curriculum |
| FGW/语义对应基础设施 | `motion_field_transfer.py` 及 Stage 1 对应模块 | Stage 1 Contact Field 不被 V7 读取 |
| Source-canonical memory | `canonical_alignment.py`, `scripts/stage2/build_v7_canonical_memory.py` | 没有显式 episode→canonical mapping 时 fail-fast |
| Target alignment/inference | `scripts/stage2/align_v7_target.py`, `infer_v7_canonical_target.py` | 使用 source Field×对应置信度；不做 online/prior 算术融合 |

当前 `V2 Joint` 和 `V6 Balanced Long` 实际上是同一个 Python 类的不同配置，V6
增加了 field sharpening、causal/drop-top intervention、consistency 和
confidence fusion 等旧实验开关。V7 不修改这些代码路径。

## 2. 数据合同和坐标构造

V7 的单个 batch 输入为：

```text
manipulated_points : [B,256,3]
manipulated_dino   : [B,256,384]
reference_points   : [B,256,3]
reference_dino     : [B,256,384]
manipulated_mask   : [B,256]
reference_mask     : [B,256]
scene_scale        : [B] 或 [B,1]
goal_pose9d        : [B,9]
trajectory_pose9d  : [B,64,9]
episode_id/object_instance_id : 字符串元数据
```

`dataset.py` 对旧缓存没有的 mask 使用全 1 兼容表示；新缓存可以保存真实有效深度/物体
mask。点云、DINO 和 mask 在训练时使用同一个随机 permutation，避免 gate 与点错位。
`schema.py` 检查固定的 256 点和可选 mask 形状。现有
`/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1` 审查结果为 179 条记录，
train/val/test=143/18/18，DINO 维度 384；旧缓存没有独立 visibility mask，也没有可靠
的 object instance ID（空值会被数据集代码回退为 episode ID）。所以该缓存可以做
episode-disjoint 训练和 smoke test，但不能声称严格的 cross-instance split。
`strict_instance_split: true` 会在读取到空 `object_instance_id` 时立即报错；正式
跨实例实验应先重新生成带实例 ID 的缓存，而不是启用 episode 回退。

对每个角色分别计算 masked 中心和共享 scene scale：

\[
c_m=\operatorname{mean}(P_m),\quad c_r=\operatorname{mean}(P_r),
\]
\[
P^m_{obj}=(P_m-c_m)/s,\quad P^r_{obj}=(P_r-c_r)/s,
\]
\[
P^m_{rel}=(P_m-c_r)/s,\quad P^r_{rel}=(P_r-c_r)/s.
\]

其中 `P_obj` 描述物体内部结构，`P_rel` 描述相对布局，避免直接使用相机绝对位置
作为捷径。Pose9D 仍为三维平移加旋转矩阵前两列的连续 6D 表示，轨迹是相对首帧的
累计对象位姿。

## 3. V7 编码计算流程

### 3.1 LocalPointEncoder：无全局旁路的局部 payload

`LocalPointEncoder` 在两种角色上各有一个独立实例：

```text
DINO: LayerNorm(384) → Linear(128) → GELU → Linear(64) → LayerNorm(64)
object XYZ:   Linear(3,32) → GELU → Linear(32,32)
relation XYZ: Linear(3,32) → GELU → Linear(32,32)
concat(64+32+32=128) → Linear(128,128) → GELU → Dropout →
Linear(128,128) → LayerNorm(128)
```

输出 `E_m,E_r:[B,256,128]`，再加入可学习 role embedding。这里不做 PointNet
global max/mean pooling，也不把对象级向量广播回点；Selector 的 hidden state 和
旧的全局 relation feature 也不会进入 `E_m/E_r`。

### 3.2 FieldSelector：只能输出标量 Field

`FieldSelector` 对 `E_m,E_r` 做两层双向 pre-norm cross-attention：

```text
LayerNorm → MultiHeadCrossAttention → residual →
LayerNorm → FFN(128→256→128) → residual
```

两个方向最后各经过 `Linear(128,64)→GELU→Linear(64,1)`，并使用 sigmoid：

\[
g_i^a=\sigma((l_i^a-b)/\tau_f)M_i^a,
\qquad a\in\{m,r\}.
\]

输出 `g_m,g_r:[B,256]` 和仅供诊断的 logits。Selector 的中间张量不在
`ContextEncoding.tokens` 中，因而不能绕过 Field 直接条件化 Goal/Trajectory。

### 3.3 Field curriculum、预算和局部连续性

训练进度 `progress∈[0,1]` 线性退火：目标选择比例从 0.50 到 0.20，Field temperature
从 1.00 到 0.40。预算损失为：

\[
L_{budget}^a=\left(\frac{\sum_iM_i^ag_i^a}{\sum_iM_i^a+\epsilon}-\rho\right)^2.
\]

在 object-centered XYZ 上用 `k=8` 近邻图，计算相邻 gate 的 L1 平滑损失
`L_smooth`。它只约束“少量且连续”，不规定杯口、杯柄等人工部位。
相同 `field_consistency_group` 的示范可用 DINO soft transport 后的 symmetric KL
作为可选一致性项；没有可靠分组时该项自动跳过。

### 3.4 GatedRelationEncoder：不可绕过的关系瓶颈

该模块重新从原始 `E_m,E_r` 建立关系。对于 manipulated→reference：

\[
L_{ij}=Q_i^mK_j^{r\top}/\sqrt{32}+\log(g_j^r+\epsilon),
\]
\[
R_i^m=g_i^m\left(E_i^m+W_O\sum_jA_{ij}V_j^r\right).
\]

reference→manipulated 对称计算。每个 block 的 FFN 后再次乘 query gate，最后使用
无 affine 的 LayerNorm 再乘 gate；因此不能写成 `E+g·update` 这种残差旁路。输出
`R_m,R_r:[B,256,128]`。

### 3.5 Field-biased FunctionalPooling：9 个 context tokens

每个角色有 4 个 learned pooling query。attention logits 加上
`log(g_i+epsilon)`，得到：

```text
Z_m : [B,4,128]
Z_r : [B,4,128]
z0  : [B,1,128] = MLP(mean(Z_m), mean(Z_r))
Z_func = concat(z0,Z_m,Z_r) : [B,9,128]
```

`z0` 同时乘两侧 gate 的平均质量；全 0 Field 时 `Z_func` 为显式零 context（数值误差
除外）。V2/V6 的 context 是 `[B,3,128]`，V7 默认是 `[B,9,128]`；Goal/Trajectory
decoder 本来就接受可变长度 memory，因此无需另建 decoder。

## 4. 两个扩散分支和损失

V7 复用现有 diffusion 主干和 public interface `compute_loss(batch)`、
`sample(batch,num_goal_samples,num_trajectory_samples)`。

* Goal 分支读取 `Z_func:[B,9,128]`，保持 9D pose（translation+rotation6D）和
  当前 scheduler/noise target，输出 `goal_noise:[B,9]`。
* Trajectory 分支读取相同 `Z_func`、9D goal 和 64 帧 pose，保持当前离散帧位置编码、
  hard-start boundary、goal conditioning 和 6 层主干，输出
  `trajectory_noise:[B,64,9]`。

联合损失为现有 Goal/Trajectory denoising loss 加：

\[
L=L_{goal}+L_{traj}+\lambda_bL_{budget}
 +\lambda_sL_{smooth}+\lambda_cL_{consistency}.
\]

每项以独立 key 写入训练日志；默认权重为
`lambda_goal=1, lambda_trajectory=1, lambda_field_budget=.02,
lambda_field_smooth=.01, lambda_field_consistency=.02`。训练基础设施保留 AdamW、EMA、
AMP、gradient clipping、best/last checkpoint、normalizer 和 resume。V7 模型在每个
epoch 开始调用 `set_training_progress`，因此 curriculum 按归一化训练进度而不是固定
epoch 名称生效。

## 5. Source-canonical memory 与目标对齐

### 5.1 记忆库

`CanonicalFieldMemory` 序列化 manipulated/reference 各自的：

```text
canonical_points [N,3]
canonical_dino   [N,384]
canonical_field_mean [N]
canonical_field_var  [N]
canonical_confidence[N]
```

`build_v7_canonical_memory.py` 要求每个 source record 同时提供
`{role}_motion_field` 和显式 `{role}_episode_to_canonical:[Ncanonical,Nepisode]`。
映射行归一化后先把每个 episode field 拉到 canonical support，再计算加权 mean/variance/
coverage。没有可靠 rigid pose、point track 或 episode→canonical 映射时脚本明确报错，
不把 episode ID 或采样顺序假装成几何对应。

### 5.2 目标实例

外部 DINO/FGW/cycle matcher 输出
`C_m,C_r:[N_source_canonical,N_target]`。`align_v7_target.py` 检查非负、有限和
每行和为 1，然后计算：

\[
\bar P_t=C P_t,\quad \bar D_t=C D_t,\quad
v_i=\max_j C_{ij},\quad g_{t,i}=F_{can,i}v_i.
\]

脚本输出对齐后的 points/DINO、source canonical field、correspondence confidence、
field gate 和有效 mask。`infer_v7_canonical_target.py` 将同一个 `g_t` 作为
`field_override` 送入同一个 V7 encoder 和同一个 Goal/Trajectory generator；不预测
target online field，也不做 `(1-alpha)online+alpha prior` 的算术融合。低置信对应可在
上层依据 confidence 阈值拒绝。

## 6. 干预、诊断与测试

`evaluate_motion_field_matrix.py` 支持 learned、uniform、roll/shuffled、complement、
drop-top、keep-top、bottom20 等 paired intervention，并在输出前对 Field mass 做归一化
以便比较 entropy/peak。完整因果路径应重新预测 Goal 再预测 Trajectory；固定 Goal
实验则只测 Field 对路径的直接影响。建议报告 Goal/Trajectory 平移旋转误差、endpoint、
first-step、Field ratio/entropy/peak/smoothness、对应置信度和 rejection rate。

新增 `tests/stage2/test_v7_functional_alignment.py` 验证：

* `[B,256,128]` local/relation、`[B,9,128]` context、`[B,9]` Goal、`[B,64,9]` Trajectory；
* 点顺序置换等变性；
* `log(g+epsilon)` 无 NaN、空 mask 处理；
* 全 0 gate 使 context 消失，uniform/shuffled 改变 context；
* LocalPointEncoder、Selector、GatedRelation、Field head 都能从 loss 获得梯度；
* correspondence 行归一化、identity mapping、canonical mean/variance；
* 旧 registry 和 V7 registry 均可构造。

本轮环境中的结果：

```text
PYTHONPATH=. /home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  -m pytest -q tests/stage2
44 passed

V7 synthetic overfit smoke（2 epochs）：val total 3.034 → 2.716
V7 pouring cache smoke（1 epoch，4 train/2 val）：checkpoint best.pt 已保存
real-cache V7 forward/sample：context [1,9,128]，goals [1,2,9]，
trajectories [1,2,1,64,9]
```

推荐命令：

```bash
# synthetic overfit
PYTHONPATH=. python scripts/stage2/train.py \
  --config configs/stage2/motion_field_v7_synthetic_overfit.yaml \
  --output-dir /tmp/lfv_v7_synthetic

# real-cache smoke
PYTHONPATH=. python scripts/stage2/train.py \
  --config configs/stage2/motion_field_v7_pouring_lfv_smoke.yaml \
  --output-dir /tmp/lfv_v7_pouring

# formal training（请先根据 GPU/步数确认）
PYTHONPATH=. python scripts/stage2/train.py \
  --config configs/stage2/motion_field_v7_functional_alignment_pouring_lfv.yaml

# source memory → target alignment → V7 inference
PYTHONPATH=. python scripts/stage2/build_v7_canonical_memory.py \
  --records RECORD_A.npz RECORD_B.npz --canonical CANONICAL.npz \
  --output source_canonical_field.npz
PYTHONPATH=. python scripts/stage2/align_v7_target.py \
  --memory source_canonical_field.npz --target TARGET.npz \
  --correspondence CORRESPONDENCE.npz --output ALIGNED.npz
PYTHONPATH=. python scripts/stage2/infer_v7_canonical_target.py \
  --checkpoint CHECKPOINT.pt --aligned-target ALIGNED.npz \
  --output V7_SAMPLES.npz --num-goals 8 --num-trajectories 2
```

## 7. 当前边界和验收结论

本轮完成的是 V7 网络、数据 mask 兼容、canonical memory/alignment 接口、干预诊断和
smoke 闭环；没有伪造现有数据中缺失的 canonical mapping，也没有把 Stage 1 Contact
Field 接入 Stage 2。现有 pouring cache 的空 `object_instance_id` 和缺失
episode→canonical correspondence 是严格跨实例实验的前置数据缺口；在补齐这些字段
之前，V7 只能报告 episode-disjoint 结果和接口级 fail-fast 行为。

V7 的四项核心验收条件对应代码中的明确边界：

1. Selector 只输出两个标量 Field；
2. Goal/Trajectory 只读取 Field-gated `Z_func`，没有未门控 global payload；
3. learned/uniform/shuffled 等 intervention 可以在同一 batch/seed 下配对比较；
4. 目标实例先通过 correspondence 拉回 source support，再复用同一个 source Field 和
   同一个运动生成器，不进行 online/prior 算术融合。
