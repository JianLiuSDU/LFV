# Stage 2 共享 Encoder 联合训练与阶段感知轨迹 Transformer 改进方案

> 日期：2026-08-07
> 状态：时间位置编码已完成代码修正；其余内容是下一轮网络设计与消融方案，尚未实现。
> 设计原则：保留当前 <code>XYZ+DINO → 三 Token Encoder → Goal Diffusion → Trajectory Diffusion Transformer</code> 主框架，不退回旧网络，也不引入繁杂人工特征。

## 1. 结论先行

当前两个 Diffusion 共用一个 Scene Encoder 并联合训练，在任务逻辑上是合理的：Goal 和 Trajectory 都需要理解被操作物体、参考物体及两者关系，共享底层感知可以减少重复计算并让两个任务使用一致的场景表征。

但当前联合训练有两个需要区分的问题：

1. **共享 Encoder 的梯度是否协调。** 当前 Encoder 梯度是 Goal loss 与 Trajectory loss 梯度的直接相加。这种更新在数学上正确，但是否合适不能通过两个标量 loss 的大小判断，必须测量 Encoder 上两路梯度的范数与夹角。
2. **Trajectory Transformer 的条件结构是否足够。** 当前三个场景 Token 与 Goal Token 只是被拼成四个静态 Memory Token，并没有先形成“针对这个 Goal，场景中哪些关系和区域更重要”的条件表征；轨迹网络也缺少显式的粗阶段潜变量。因此终态容易学好，中间轨迹仍容易退化成低频弧线。

推荐保持共享 Encoder 和两个 Diffusion，按以下顺序改进：

1. 已完成：修正离散轨迹帧位置编码；
2. 保持联合训练，但增加 Goal/Trajectory 总损失权重配置和共享 Encoder 梯度诊断；
3. 在 Trajectory Diffusion 前加入一个很小的 Goal-conditioned Context Mixer；
4. 从 Goal-conditioned context 生成 4 个有顺序的 latent phase tokens；
5. 把当前轨迹 Block 改成“局部时间注意力 + 阶段注意力 + 场景关系注意力 + 稀疏全局注意力”；
6. 只有前三步消融仍不足时，才让 Encoder 额外保留少量局部点 Token。

## 2. 已完成的时间位置编码修正

### 2.1 原问题

原实现把 64 个轨迹位置写成：

~~~text
progress = linspace(0,1,64)
position_embedding = SinusoidalEmbedding(progress)
~~~

由于正弦编码最高频率只有 1，整条轨迹最多只覆盖 1 rad，其余频带变化更小。实测 128 维编码的 participation rank 约为 1.20，几乎退化成一个缓慢变化的单维信号。

### 2.2 新实现

新训练配置使用绝对离散帧索引：

~~~text
with fixed start token:
    positions = [0,1,2,...,63]

without fixed start token:
    positions = [1,2,...,63]
~~~

然后仍使用相同 SinusoidalEmbedding。这样不增加可训练参数，也不改变 Tensor Shape，但能让不同帧获得明显可分辨的时间基函数。64 个离散位置的 participation rank 约为 16.90。

### 2.3 Checkpoint 兼容

代码新增显式配置：

~~~yaml
model:
  trajectory_position_encoding: discrete_sinusoidal
~~~

支持两种模式：

| 模式 | 用途 |
|---|---|
| <code>discrete_sinusoidal</code> | 新训练，使用帧索引 0–63 |
| <code>legacy_normalized_sinusoidal</code> | 复现旧 checkpoint 的 0–1 编码 |

旧 checkpoint 的保存配置中没有该字段。加载这类 checkpoint 时，代码自动选择 legacy 模式，避免在不报错的情况下改变其推理函数。所有新训练 YAML 已显式选择 discrete 模式。

这意味着：

- 旧 checkpoint 仍能按旧方式复现；
- 时间编码修正的效果必须通过重新训练验证；
- 不能加载旧权重、强制切到新位置编码后直接把结果当作修复效果。

### 2.4 已加入的测试

新增测试验证：

1. 固定起点模式的位置严格为 0、1、2、…；
2. 无固定起点模式从位置 1 开始；
3. 64 个离散正弦编码不会退化；
4. legacy 模式保持原来的 0–1 行为；
5. 不含版本字段的旧 checkpoint 配置自动使用 legacy；
6. 非法模式立即报错。

相关 Stage2 forward、sampling 和 checkpoint 测试一并通过。

## 3. 两个 Diffusion 共用 Encoder 时，梯度怎样更新

### 3.1 当前参数划分

记：

~~~text
Scene Encoder parameters      = theta
Goal Diffusion parameters     = phi
Trajectory Diffusion params   = psi
~~~

输入场景为 <code>x</code>：

~~~text
C = Encoder_theta(x)                 # [B,3,128]
goal_hat = GoalDiffusion_phi(C)
traj_hat = TrajectoryDiffusion_psi(C, goal_condition)
~~~

训练时 Trajectory 使用扰动后的 GT Goal，而不是 Goal Diffusion 的采样结果。

### 3.2 当前总损失

当前代码等价于：

~~~text
L_total = L_goal + L_trajectory
~~~

每个子任务内部又有：

~~~text
L_goal =
    L_goal_diffusion
  + L_goal_translation
  + 0.5 × L_goal_rotation

L_trajectory =
    L_trajectory_diffusion
  + L_translation
  + 0.5 × L_rotation
  + 0.5 × L_velocity
  + 0.1 × L_acceleration
  + 2.0 × L_start
  + L_endpoint
~~~

### 3.3 一次 backward 后各模块得到的梯度

~~~mermaid
flowchart TB
    X["XYZ+DINO scene"] --> E["Shared Encoder theta"]
    E --> C["3 context tokens"]
    C --> G["Goal Diffusion phi"]
    C --> T["Trajectory Diffusion psi"]
    YG["GT Goal"] --> LG["Goal loss"]
    G --> LG
    YT["GT trajectory + perturbed GT Goal"] --> LT["Trajectory loss"]
    T --> LT
    LG --> SUM["lambda_goal L_goal<br/>+ lambda_traj L_traj"]
    LT --> SUM
    SUM --> GE["Encoder gradient<br/>lambda_goal g_goal + lambda_traj g_traj"]
    SUM --> GG["Goal gradient<br/>only from Goal loss"]
    SUM --> GT["Trajectory gradient<br/>only from Trajectory loss"]
~~~

具体为：

~~~text
gradient on shared Encoder:
    grad_theta =
        lambda_goal × dL_goal/dtheta
      + lambda_traj × dL_trajectory/dtheta

gradient on Goal Diffusion:
    grad_phi =
        lambda_goal × dL_goal/dphi

gradient on Trajectory Diffusion:
    grad_psi =
        lambda_traj × dL_trajectory/dpsi
~~~

当前 <code>lambda_goal=lambda_traj=1</code>。

因为 Trajectory 条件中的 Goal 来自 GT，而不是 Goal Decoder，所以：

~~~text
dL_trajectory/dphi = 0
~~~

也就是说，Trajectory loss 不会直接更新 Goal Decoder，只会更新共享 Encoder 和 Trajectory Decoder。这是一个稳定、清晰的训练图。

### 3.4 实际 optimizer 更新

当前单个 batch 的正确更新方式是：

~~~text
optimizer.zero_grad()

with AMP:
    context = encoder(batch)
    L_goal = goal_loss(context)
    L_traj = trajectory_loss(context)
    L_total = lambda_goal × L_goal + lambda_traj × L_traj

scaler.scale(L_total).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(all_model_parameters)
scaler.step(optimizer)
scaler.update()

EMA.update(model)
~~~

共享 Encoder 应在两项损失相加后只更新一次。不建议在同一个 batch 中先用 Goal loss 更新 Encoder，再用 Trajectory loss 更新同一个 Encoder，因为第二次更新看到的参数已经变化，任务顺序会引入额外偏置，并且需要复杂地处理计算图。

## 4. 共用 Encoder 联合训练是否合理

### 4.1 合理之处

Goal 和 Trajectory 共享以下认知：

- 哪一组点属于 manipulated object；
- 哪一组点属于 reference object；
- 两者的空间关系；
- DINO 语义对应；
- 操作物体相对于参考物体的任务相关方向。

让两个任务完全使用独立 Encoder 会重复学习这些信息，也可能使 Goal 与 Trajectory 对同一场景形成不一致解释。因此共享基础 Encoder 是合理的，并且应该保留。

### 4.2 当前可能出现的梯度问题

#### 问题一：梯度冲突

定义共享 Encoder 上两路梯度：

~~~text
g_goal = dL_goal/dtheta
g_traj = dL_trajectory/dtheta

cosine =
    dot(g_goal,g_traj)
    / (norm(g_goal) × norm(g_traj))
~~~

解释：

| 梯度关系 | 含义 |
|---|---|
| cosine > 0 | 两个任务在推动 Encoder 学相近方向 |
| cosine ≈ 0 | 两个任务大体独立，共享通常没有问题 |
| cosine < 0 | 两个任务对 Encoder 存在冲突 |
| norm_goal 远大于 norm_traj | Encoder 更容易变成终态特征编码器 |
| norm_traj 远大于 norm_goal | Goal 表征可能被轨迹重构目标扰乱 |

当前日志只有标量 loss，没有记录这两个梯度范数和夹角。因此目前不能仅凭 <code>L_goal</code> 小于或大于 <code>L_trajectory</code> 判断谁主导 Encoder。Loss 数值和参数梯度大小不是同一件事。

#### 问题二：Trajectory 对 Goal 的捷径依赖

Trajectory Decoder 的 Memory 中有一个非常直接的 Goal Pose Token。对于训练数据，给定起点和终点后，一条平均弧线已经能显著降低逐帧误差。因此 Decoder 可能主要读取 Goal Token，而较少读取三个 Scene Token。

结果是：

~~~text
Goal token          → 决定大致终点与弧线方向
3 scene tokens      → 只做小幅修正
local scene details → Encoder 池化时已经丢失
~~~

这不是共享 Encoder 更新公式错误，而是 Trajectory Decoder 使用共享特征的方式不够强。

#### 问题三：简单任务更早收敛

Goal 是单个 9D 输出，Trajectory 是 63×9 输出。Goal 通常更快找到稳定映射。联合训练后期若 Goal loss 已经平台化，它仍持续向共享 Encoder 提供梯度，可能限制 Encoder 为中间路径学习新的表示。

但这只是一种可能性，必须通过梯度诊断和 loss-weight 消融验证，不能直接假定。

### 4.3 最小修改建议

第一轮不拆 Encoder，也不立即引入复杂多任务优化器，只做：

1. 在配置中增加 <code>goal_total_weight</code> 和 <code>trajectory_total_weight</code>；
2. 初始仍使用 1:1，作为离散位置编码修复后的基线；
3. 每隔固定 step 记录共享 Encoder 上：
   - <code>norm(g_goal)</code>；
   - <code>norm(g_traj)</code>；
   - 两者 cosine similarity；
4. 若 Goal 梯度长期超过 Trajectory 梯度 2–3 倍，再尝试把 Goal 总权重降到 0.25–0.5；
5. 若 cosine 长期明显为负，先增加两个很小的 task-specific adapters，再考虑 PCGrad；
6. 不根据单次 batch 的梯度动态切换权重，先看滑动平均。

建议的小 Adapter 为：

~~~text
C_base = SharedEncoder(scene)

C_goal = C_base + MLP_goal(LayerNorm(C_base))
C_traj = C_base + MLP_traj(LayerNorm(C_base))
~~~

它保留绝大部分共享参数，只给两个任务一个很小的特征调整空间。相比直接复制两个 Encoder，修改和计算量都很小。

## 5. 三个场景 Token 是否应该与 Goal 交互

### 5.1 当前实际情况

当前只执行：

~~~text
memory = concat(
    three_scene_tokens,
    goal_token,
)
~~~

随后每个 trajectory token 直接对这四个 Memory Token 做 Cross-Attention。三个 scene tokens 本身不会因为 Goal 不同而改变。

对于同一场景的 K 个 Goal：

~~~text
scene token 0,1,2  完全相同
goal token          随 Goal 改变
~~~

这把“结合目标解释场景”的责任全部留给 64 个带噪轨迹 Token。高噪声阶段的 trajectory query 本身不稳定，因此它未必能可靠完成这种交互。

### 5.2 应该交互，但不应放回基础 Encoder 内

建议保留基础 Encoder：

~~~text
scene → Shared Encoder → C_base [B,3,128]
~~~

Goal 生成后，在 Trajectory Diffusion 前增加：

~~~text
C_goal = GoalConditionedContextMixer(C_base, goal)
~~~

原因是：

- Goal Decoder 需要读取不依赖候选 Goal 的基础场景表示；
- Trajectory Decoder 需要读取“针对当前 Goal 重新解释过”的场景表示；
- 一个场景有 K 个 Goal 时，应产生 K 组不同的 goal-conditioned context；
- 这样不会形成 Goal Decoder 依赖自己输出的循环。

### 5.3 不要只做 scene-to-single-goal cross-attention

如果 K/V 只有一个 Goal Token：

~~~text
CrossAttention(Q=scene tokens, K=one goal, V=one goal)
~~~

Softmax 只有一个元素，权重必然为 1。三个 scene query 最终拿到几乎相同的 Goal value，这种“Cross-Attention”在数学上是退化的，不能真正学习不同场景 Token 如何选择 Goal 信息。

推荐使用一个 2 层 Joint Context Mixer：

~~~mermaid
flowchart LR
    C["3 base scene tokens<br/>B×3×128"] --> J["Concatenate"]
    G["Goal pose<br/>MLP 9→128"] --> J
    J --> T["2-layer self-attention<br/>over 4 tokens"]
    T --> O["Goal-conditioned context<br/>B×4×128"]
~~~

输入：

~~~text
[initial, M→R, R→M, goal]
 + token-type embeddings
~~~

每层：

~~~text
LayerNorm → 4-token self-attention → residual
LayerNorm → FFN → residual
~~~

Self-Attention 有四个 K/V，所以三个场景 Token 可以根据 Goal 和其他关系 Token 形成不同更新；Goal Token 也能读取场景关系。

## 6. 在 Transformer 前加入 latent phase tokens

### 6.1 为什么需要阶段 Token

正确的离散位置编码只能告诉网络“这是第几帧”，但不能自动告诉它：

~~~text
这些相邻帧属于同一个粗动作阶段；
这一阶段应该重点看哪种场景关系；
阶段之间何时发生信息切换。
~~~

当前每帧都直接访问同一组静态 Memory，因此很容易学成连续、均匀变化的弧线。

### 6.2 Phase Token Generator

建议使用 <code>P=4</code> 个 learned phase queries：

~~~text
phase_queries [1,4,128]
 + ordered phase embeddings
 → cross-attend Goal-conditioned context [B,4,128]
 → phase self-attention
 → phase_tokens [B,4,128]
~~~

~~~mermaid
flowchart LR
    Q["4 learned ordered<br/>phase queries"] --> CA["Cross-attention"]
    C["Goal-conditioned context<br/>B×4×128"] --> CA
    CA --> SA["Phase self-attention"]
    SA --> P["4 latent phase tokens<br/>B×4×128"]
~~~

这些 Token 不需要人工标注为“抬升、平移、接近、倾倒”。它们只是四个有顺序的潜在阶段槽位，具体语义由数据学习。

### 6.3 防止 Phase Token 全部学成相同内容

只放四个 learned queries 仍可能发生 phase collapse。第一版建议加入固定的时间中心：

~~~text
mu = [0.0, 1/3, 2/3, 1.0]
tau_k = k / 63

phase_bias(k,p) =
    -(tau_k - mu_p)² / (2 sigma²)
~~~

这个 bias 只约束“早期帧更容易读取早期 Phase Token，后期帧更容易读取后期 Phase Token”，不定义每个阶段的人工语义。

也可以让中心可学习，但应使用单调参数化保证顺序：

~~~text
delta_p = softplus(raw_delta_p)
mu_p = cumulative_sum(delta_p) / total_sum(delta)
~~~

第一版建议固定均匀中心，减少变量。

## 7. 推荐的 Stage-aware Trajectory Transformer

### 7.1 总体结构

~~~mermaid
flowchart TD
    S["XYZ+DINO"] --> E["Shared 3-token Encoder"]
    E --> CB["Base context B×3×128"]
    CB --> GD["Goal Diffusion"]
    GD --> G["K Goal poses"]
    CB --> CM["2-layer Goal-conditioned<br/>Context Mixer"]
    G --> CM
    CM --> GC["Per-goal context<br/>B×K×4×128"]
    GC --> PG["4-token Phase Generator"]
    PG --> PT["Per-goal phase tokens<br/>B×K×4×128"]
    N["63 noisy Pose tokens<br/>with discrete positions"] --> TD["6 Stage-aware<br/>Trajectory Blocks"]
    GC --> TD
    PT --> TD
    TD --> O["Clean trajectory<br/>B×K×M×63×9"]
~~~

### 7.2 推荐 Block

保留 6 层、hidden 128、4 heads，不先扩大模型。每个 Block 改为：

~~~mermaid
flowchart TB
    X["Trajectory tokens"] --> L["AdaLN + local temporal attention<br/>window 7 or dilated window"]
    L --> P["AdaLN + trajectory-to-phase attention<br/>with monotonic phase bias"]
    P --> C["AdaLN + cross-attention<br/>to goal-conditioned context"]
    C --> G{"Global layer?"}
    G -->|layers 2,4,6| A["AdaLN + full self-attention"]
    G -->|layers 1,3,5| S["skip global attention"]
    A --> F["AdaLN + FFN"]
    S --> F
    F --> O["Block output"]
~~~

各子层作用：

| 子层 | 解决的问题 |
|---|---|
| corrected discrete position | 明确区分 64 个帧位置 |
| local/window attention | 保留邻近运动连续性，不让所有帧过早全局平均 |
| phase cross-attention | 让帧读取对应粗阶段的目标与场景摘要 |
| goal-conditioned context attention | 让每个 Goal 对场景关系产生不同解释 |
| sparse global attention | 在部分层允许远距离帧协调起终点和整体动作 |
| FFN | 对融合后的每帧表示做非线性更新 |

### 7.3 为什么使用局部与全局交替

当前每一层都做 Full Self-Attention：

~~~text
all frames ↔ all frames, repeated 6 times
~~~

在弱阶段信号下，它容易快速把信息传播到全序列，形成低频平均。全部改成局部注意力又会丢失终点协调。因此推荐：

~~~text
layer 1: local
layer 2: local + global
layer 3: local
layer 4: local + global
layer 5: local
layer 6: local + global
~~~

这仍然是 Transformer，只是给时间结构增加合理的层级归纳偏置。

### 7.4 是否还需要当前 Temporal Conv

两种可选方式：

1. 保留 <code>Conv1d(k=3)</code>，再增加 window attention；
2. 用带 relative temporal bias 的 window attention 替代 Conv。

第一版建议保留 Conv，因为代码改动小、已有 checkpoint 结构容易对照。若 window attention 已明显改善，再消融 Conv 是否冗余。

## 8. 三个 Encoder Token 是否够用

### 8.1 当前阶段先保留

三个 Token 对 Goal 有效，当前首要问题已经明确包含时间编码和轨迹条件结构。因此第一轮不应同时重写 Encoder。

推荐先验证：

~~~text
3 base tokens
 + goal-conditioned context mixer
 + 4 phase tokens
 + improved temporal Transformer
~~~

如果训练集轨迹折点和阶段仍无法拟合，再判定三 Token 确实构成不可逆信息瓶颈。

### 8.2 后续最小扩展

若需要增加局部信息，不恢复复杂人工 relation features，只让两个 PointNet 分支各保留少量 learned pooled tokens：

~~~text
global tokens:
    initial, M→R, R→M                # existing 3

local tokens:
    8 manipulated pooled tokens
    8 reference pooled tokens
~~~

Trajectory 的 Goal-conditioned Context Mixer 可以读取这些局部 Token；Goal Decoder仍可以只读三个全局 Token，避免让简单任务变复杂。

是否需要局部 Token 的判断标准不是 Goal 指标，而是：

- GT Goal 条件下的中间路径误差；
- 阶段转折位置；
- 路径与 GT 的曲率/方向变化；
- 模拟中的碰撞和净空。

## 9. 联合训练在新结构中的梯度路径

新增：

~~~text
Goal-conditioned Context Mixer parameters = eta
Phase Generator parameters                = rho
Stage-aware Trajectory Transformer        = psi
~~~

训练图为：

~~~text
C_base = Encoder_theta(scene)

L_goal =
    GoalDiffusion_phi(C_base, GT_goal)

C_goal =
    ContextMixer_eta(C_base, perturbed_GT_goal)

phase_tokens =
    PhaseGenerator_rho(C_goal)

L_traj =
    TrajectoryTransformer_psi(
        noisy_trajectory,
        C_goal,
        phase_tokens,
    )
~~~

梯度：

~~~text
theta receives: lambda_goal g_goal + lambda_traj g_traj
phi   receives: lambda_goal g_goal_decoder
eta   receives: lambda_traj g_context_mixer
rho   receives: lambda_traj g_phase_generator
psi   receives: lambda_traj g_trajectory
~~~

这种划分保留共享感知，但 Goal-conditioned Mixer、Phase Generator 和 Trajectory Transformer 都只服务轨迹，不会反过来增加 Goal Decoder 的负担。

## 10. 对损失的建议

第一轮位置编码和网络结构消融中，不建议同时大改 diffusion target 或轨迹表示：

- 保持累计 Pose9D；
- 保持 clean-sample prediction；
- 保持当前 translation、SO(3)、velocity、start、endpoint loss；
- 暂时保留 acceleration loss，但单独记录消融；
- 新增 Goal/Trajectory 顶层权重，而不是继续堆更多人工几何 loss。

可以增加一个与阶段结构直接相关、但不依赖人工阶段标注的多尺度轨迹损失：

~~~text
L_multiscale =
    error at 64 frames
  + w2 × error at pooled 32 frames
  + w4 × error at pooled 16 frames
  + w8 × error at pooled 8 frames
~~~

它让网络同时看到局部轨迹和粗动作轮廓，比单独增大 acceleration loss 更贴近阶段建模。但这应放在 Phase Token 消融之后，避免无法区分结构收益来自哪里。

## 11. 推荐的实施顺序

### V3.1：已完成

~~~text
discrete sinusoidal frame positions
legacy checkpoint compatibility
position encoding unit tests
~~~

需要重新训练，与 legacy encoding 使用同一数据、split、seed 和训练步数对比。

### V3.2：最小共享训练诊断

~~~text
configurable lambda_goal / lambda_traj
encoder gradient norms
encoder gradient cosine
attention usage statistics for four memory tokens
~~~

这一版不改网络，只回答联合训练是否真的发生梯度冲突，以及 Trajectory 是否忽略 Scene Token。

### V3.3：Goal-conditioned context

~~~text
2-layer Context Mixer over:
[initial, M→R, R→M, goal]
~~~

验证同一场景不同 Goal 是否得到不同的场景关系 Token。

### V4：阶段感知 Transformer

~~~text
4 latent phase tokens
monotonic frame-to-phase bias
local/global alternating trajectory blocks
~~~

保持 hidden 128、heads 4、layers 6。

### V4.1：可选局部几何 Token

只有 V4 在训练集上仍不能恢复 GT 阶段时，增加：

~~~text
8 manipulated local tokens
8 reference local tokens
~~~

不先增加法向、碰撞、语言或人工 waypoint。

## 12. 消融实验矩阵

| 实验 | Position | Context Mixer | Phase tokens | Local/global attention | 目的 |
|---|---|---|---|---|---|
| A0 | legacy 0–1 | 无 | 无 | current full | 旧基线 |
| A1 | discrete | 无 | 无 | current full | 单独验证时间编码 |
| A2 | discrete | 有 | 无 | current full | 验证 Goal-scene 交互 |
| A3 | discrete | 有 | 4 | current full | 验证 Phase Token |
| A4 | discrete | 有 | 4 | alternating | 完整 Stage-aware Transformer |
| A5 | discrete | 有 | 4 | alternating + local scene tokens | 验证三 Token 信息瓶颈 |

每个实验必须同时报告：

- Goal translation / rotation；
- Trajectory conditioned on predicted Goal；
- Trajectory conditioned on GT Goal；
- 第一步位移及相对训练 p95；
- 全程 translation / rotation；
- endpoint error；
- 路径总长和路径效率；
- 曲率或二阶差分曲线与 GT 的相关性；
- 主要方向转折发生帧；
- DTW trajectory error；
- 同一 Goal 下 sample diversity；
- Best-of-K 相对 Top-1 的改善。

## 13. 最终回答

### 两个 Diffusion 共用 Encoder 是否合理

合理，建议保留。当前一次计算三个场景 Token，两项 loss 相加后一次 backward，Encoder 获得两路梯度之和；Goal 和 Trajectory Decoder 分别只接收自己的梯度。这是正确的联合训练方式。

需要补的是梯度范数和夹角诊断，而不是立即拆成两个 Encoder。若确实存在冲突，优先使用顶层 loss weight 和小型 task adapter。

### 三个 Token 是否应与 Goal 交互

对 Goal Decoder 不需要预先交互，因为 Goal 尚未生成；对 Trajectory Decoder 应该交互。最佳位置是在 Goal 生成之后、Trajectory Transformer 之前加入一个 2 层、4 Token 的 Goal-conditioned Context Mixer。不要用三个 scene query 对单个 Goal K/V 做退化 Cross-Attention。

### 如何改善 Transformer 的阶段理解

正确位置编码只是第一步。建议增加四个有顺序但无人工语义标签的 latent phase tokens，并让每个轨迹帧通过单调时间 bias 读取相应阶段；同时让轨迹 Block 在局部时间注意力和稀疏全局注意力之间交替。这样保持现有 Transformer 主体，却能显式表达“帧位置—粗阶段—场景关系—终态目标”四层条件。

### 当前最值得实现的下一版

~~~text
Shared 3-token Encoder
 → Goal Diffusion
 → 2-layer Goal-conditioned Context Mixer
 → 4 latent Phase Tokens
 → 6-layer local/global Stage-aware Trajectory Transformer
~~~

这一结构比继续增加平滑损失更直接地针对当前问题，也不需要退回旧方法。

## 14. 代码位置

- 时间编码修正：<code>lfv/models/functional_motion_generation/trajectory/decoder.py</code>
- 配置与旧 checkpoint 兼容：<code>lfv/models/functional_motion_generation/loading.py</code>
- 模型参数传递：<code>lfv/models/functional_motion_generation/system.py</code>
- 新训练配置：<code>configs/stage2/*.yaml</code>
- 时间编码测试：<code>tests/stage2/test_trajectory_position_encoding.py</code>
- 当前联合损失：<code>lfv/models/functional_motion_generation/system.py</code>
- 当前训练器更新：<code>lfv/training/functional_motion/trainer.py</code>
