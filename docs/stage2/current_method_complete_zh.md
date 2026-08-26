# LFV 当前方法完整说明：语义—结构任务场与分层运动扩散

> 版本：Stage 2 V2 Joint Motion Functional Field
>
> 适用代码分支：`stage2/motion-functional-field`
>
> 当前模型版本：`stage2-motion-field-v2.2-trained`
>
> 本文描述当前已经实现、训练和验证过的项目方法。它同时说明项目级的
> Stage 1 接触迁移与 Stage 2 目标/轨迹生成之间的关系；其中 Stage 2 的
> Motion Functional Field 是当前 V2 的正式模型输出，Stage 1 的 Contact
> Field 不作为 Stage 2 网络输入。

---

## 1. 方法要解决的问题

LFV 将“从人类示范中理解应该接触哪里”和“把被操作物体完成到什么状态、沿什么路径运动”分成两个相互衔接但监督目标不同的阶段：

```text
人类示范 / 源图像
      │
      ├── Stage 1：接触语义迁移 → 目标 Contact Field → 抓取候选
      │
      └── Stage 2：目标状态与运动学习 → Goal Pose + 64-step Trajectory

目标 Contact Field + 目标/轨迹候选
      │
      ▼
抓取、固定物体—夹爪关系、IK/碰撞筛选、执行
```

核心设计原则有三点：

1. 用统一的逐点 `XYZ + DINOv2` 表示承载语义和几何，不把 Contact heat 强行
   当作运动网络的初始观测；
2. 用两个有角色区别的对象编码器和双向 cross-attention 建立“被操作物体—参考
   物体”的关系；
3. 在 Stage 2 中把运动相关性作为网络内部真正参与计算的显式 Motion Functional
   Field，而不是把 attention map 在事后当作解释图。

当前倒水任务的被操作物体是杯子，参考物体是碗。对其他任务，只需替换数据适配器、
mask 和配置中的任务路径，模型接口保持不变。

---

## 2. 项目级完整流程

### 2.1 训练阶段

```text
RGB-D 首帧 + manipulated/reference masks
        │
        ├─ 有效深度过滤
        ├─ 每个角色确定性采样 256 个像素
        ├─ 同一索引反投影 XYZ，并采样同位置 DINOv2 descriptor
        └─ 保存离线缓存
                  │
                  ▼
      双对象 Scene Encoder
        │       │       │
        │       │       └─ reference-query-manipulated relation token
        │       └───────── manipulated-query-reference relation token
        └───────────────── initial scene token
                  │
                  ▼
      Joint Motion Functional Field J
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
  Goal Pose Diffusion   Trajectory Diffusion
  输出 K_g 个 9D goal   每个 goal 输出 K_t 条 64×9D trajectory
        └─────────┬──────────┘
                  ▼
       训练损失和 EMA checkpoint
```

训练时 Goal 和 Trajectory 两个分支共享一个 Scene Encoder。联合训练损失的梯度同时
更新两个扩散分支和 encoder；Motion Field 没有单独的人工标注，而是受到 Goal/Trajectory
任务损失的隐式监督。

### 2.2 推理和执行阶段

```text
一张目标场景 RGB-D
      │
      ├─ mask、采样、DINO、反投影
      ├─ encoder 得到三个 context token 和两个 motion field
      ├─ DDIM 生成 K_g 个终态候选
      ├─ 对每个终态生成 K_t 条 64 帧轨迹
      ├─ 轨迹和终态从 local/camera frame 变回仿真或机器人 world frame
      ├─ Stage 1 Contact Field 指导抓取候选选择
      ├─ 固定 object-to-TCP attachment
      ├─ IK、碰撞、工作空间和夹爪约束筛选
      └─ 执行并记录 RGB/坐标系/轨迹视频
```

Stage 2 只生成物体运动；GraspNet、夹爪闭合、IK 和控制不属于扩散网络本身。这样的
模块边界使得抓取候选和运动候选可以独立替换、排序和回放。

---

## 3. 数据表示与坐标约定

### 3.1 固定数据协议

当前 `pouring_lfv` 派生缓存为：

```text
有效 episode：179
train / val / test：143 / 18 / 18
随机种子：42
manipulated_points：256×3，float32
reference_points：256×3，float32
manipulated_dino：256×384，float16 离线保存
reference_dino：256×384，float16 离线保存
goal_pose9d：9
trajectory_pose9d：64×9
```

源数据当前没有可靠的 `object_instance_id`。因此当前 split 是 episode-disjoint
baseline，而不是严格 object-instance-disjoint 泛化测试。论文中必须明确这一点。

### 3.2 点云和 DINO 的同步采样

每个角色在 mask 内执行：

1. 过滤无效深度以及不在工作范围内的深度值；
2. 用确定性 image-space FPS 选出恰好 256 个互不重复像素；
3. 用同一个像素索引从 depth 反投影得到 XYZ；
4. 在同一个像素位置双线性采样 DINOv2 patch descriptor；
5. 对逐点 DINO descriptor 做 L2 normalization；
6. 训练时只对 `(point_i, dino_i)` 做联合随机置换。

因此，网络看到的第 `i` 个 DINO 特征一定对应第 `i` 个三维点，不存在点云和语义
特征独立打乱造成的伪对应。

### 3.3 局部坐标系与 Pose9D

以 manipulated object 点云质心 (c) 为中心，尺度记为 (s)：

\[
p_{local}=(p_{camera}-c)/s.
\]

轨迹标签是相对于首帧的累计刚体变换 (T_{0\rightarrow k}=[R_k,t_k])，不是相邻
帧增量 (T_{k-1\rightarrow k})。每一帧使用：

```text
Pose9D = [tx, ty, tz, r00, r10, r20, r01, r11, r21]
```

即三维平移加旋转矩阵前两列的连续 6D 表示。解码时通过 Gram–Schmidt 恢复
(SO(3))。平移只使用 train split 统计得到的均值和标准差归一化；旋转不进行
普通均值方差归一化。

推理时先在 local/camera frame 得到 (hat T_{0\rightarrow k})，再用完整的刚体
矩阵复合恢复 world pose。不能只对 xyz 做逐帧普通加法，也不能把已经是累计变换的
64 帧再次逐帧累乘。

---

## 4. Stage 1：Contact Field 的语义—结构迁移

Stage 1 的目的不是预测物体运动，而是回答“目标实例上哪些可见区域与示范中的接触
区域对应”。它输出连续 Contact Field，后续用于抓取候选生成和排序。

### 4.1 源接触场

从源视频中选择手—物接触证据较强的帧，依据手部遮挡、手部关键点和物体 mask 在
物体表面构造连续热力 (A_s\in[0,1])。多帧可以在同一示范内对齐后融合，但第一版
可以先使用一个清晰源帧。

需要强调：源 Contact Field 是接触监督，不是 Stage 2 的 Motion Functional Field。
前者描述“哪里适合抓取/接触”，后者描述“哪些点对完成目标运动更相关”。两者语义
不同，不能在文中混称为同一个场。

### 4.2 DINO 语义对应

源图和目标图使用同一包围框留边、缩放和填充规则。冻结的 DINO/DINOv2 提取稠密
patch descriptor，并逐向量 L2 normalize。源 mask 和目标 mask 被缩放到特征网格。

在源热力高于阈值的区域执行热力加权 K-means，得到源接触原型
(z_k^s) 及质量权重：

\[
\omega_k=\frac{\sum_{i\in C_k}A_s(i)}{\sum_i A_s(i)}.
\]

目标前景区域执行较密集的 K-means 过分割。计算源原型与目标区域原型的 cosine
similarity，沿目标区域做 temperature softmax 得到正向匹配。再让目标原型反向匹配
到源物体全部前景 descriptor，并用归一化源热力计算反向落回接触区的概率。正向投票
与反向验证相乘得到目标区域分数 (H_j)，再分配回目标区域内所有 patch 并插值回原图。

### 4.3 FGW 结构对应

AffCorrs 式 DINO 对应擅长确定“是不是同一个功能部件”，但容易把源部件中的局部
接触热力扩展到整个目标部件。当前增强思路是在完整可见源/目标功能部件点云上加入
Fused Gromov-Wasserstein：

1. 对整个源/目标部件点云下采样到约 256–512 点；
2. 将 DINO descriptor 投影到三维点；
3. 构造跨对象语义代价 (C_{ij}^{sem}=1-\cos(f_i^s,f_j^t))；
4. 在源、目标点云内建立 kNN 图并计算归一化 geodesic 距离 (D_s,D_t)；
5. 用 POT 的 entropic FGW 求软传输矩阵 (T)；
6. 不重新二值分割，而是直接传输源 Contact 概率：

\[
H_t(j)=\frac{\sum_iT_{ij}A_s(i)}{\sum_iT_{ij}+\epsilon}.
\]

该模块的作用是保留部件内部相对结构位置，例如“把手中央高、两端低”，而不依赖
人工 PCA、OBB、左右方向或杯口检测。

### 4.4 Stage 1 的可见性边界

单视角 RGB-D 只能产生可观测表面的 Contact Field。仿真中可以使用完整点云把可见
热力传播到遮挡侧，再交给 GraspNet 生成跨两侧抓取；真实机器人则需要额外的离线
多视角重建、CAD/SAM3D 补全或安全抓取先验。论文中应把“可见热力迁移”和“完整几何
上的抓取实例化”分为两个步骤，不能声称单视角本身恢复了完整接触表面。

---

## 5. Stage 2 Scene Encoder 与 Motion Functional Field

### 5.1 输入和角色不对称性

输入为：

```text
manipulated_points [B,256,3]
manipulated_dino   [B,256,384]
reference_points   [B,256,3]
reference_dino     [B,256,384]
```

DINO projector 在两侧共享，以保持语义空间一致；XYZ projector 和 PointNet 分支在
两侧独立，以允许网络学习“被移动的杯子”和“承接的碗”不同的角色特征。

### 5.2 初始逐点编码

共享 DINO projector：

```text
LayerNorm(384) → Linear(384,256) → GELU → Linear(256,64)
```

两侧 XYZ projector：

```text
Linear(3,64) → GELU
```

每个点拼接成 128 维输入，进入两个独立 PointNet Branch，得到：

\[
F_m,F_r\in\mathbb R^{B\times256\times128},
\qquad
g_m,g_r\in\mathbb R^{B\times128}.
\]

PointNet 的 max pooling 保证对点顺序不敏感；因为 point 和 DINO 始终联合置换，
语义—几何对应不会被破坏。

### 5.3 双向关系编码

被操作物体查询参考物体：

\[
\tilde F_m=\operatorname{MHA}(Q=F_m,K=F_r,V=F_r).
\]

将 (F_m) 和 (	ilde F_m) 拼接后经过 fusion MLP，得到逐点关系特征
(U_m(i))。反向分支独立计算 (U_r(j))：

\[
\tilde F_r=\operatorname{MHA}(Q=F_r,K=F_m,V=F_m).
\]

在没有 Motion Field 的旧模式中，relation token 使用逐点 max pooling。V1 将其替换
为两个独立 relevance head，但实验发现仅靠独立场仍容易通过 global token 绕过场。

### 5.4 V2 Joint Motion Functional Field

V2 为两侧逐点关系特征预测 logits (a_i,b_j)。双向 attention 提供互惠兼容性：

\[
C_{ij}=\frac{1}{2}\left(
\log A_{m\rightarrow r,ij}+\log A_{r\rightarrow m,ji}
\right).
\]

联合关系 logits 为：

\[
L_{ij}=a_i+b_j+\lambda_pC_{ij},
\]

并在所有点对上做带温度的二维 softmax：

\[
J_{ij}=\operatorname{Softmax}_{i,j}(L_{ij}/\tau).
\]

当前配置为 (	au=0.25,\lambda_p=0.25)。两个对象的运动场不是两个互不相关的
heatmap，而是同一个联合关系分布的边缘：

\[
H_m(i)=\sum_jJ_{ij},\qquad H_r(j)=\sum_iJ_{ij}.
\]

两个 relation token 使用对应边缘进行加权：

\[
z_{m\leftarrow r}=\sum_iH_m(i)U_m(i),
\qquad
z_{r\leftarrow m}=\sum_jH_r(j)U_r(j).
\]

V2 的 initial token 也不再使用未筛选的 global bypass，而是由两个 field-weighted
point summary 融合得到：

\[
z_{init}=MLP\left(
\left[\sum_iH_m(i)F_m(i),
\sum_jH_r(j)F_r(j)\right]
\right).
\]

最终 encoder 输出：

```text
[z_init, z_m<-r, z_r<-m]  # [B,3,128]
```

每个 token 加上角色 type embedding 后送入两个扩散分支。

### 5.5 Motion Field 的监督来源

当前 Motion Functional Field 没有人工逐点标签。它是由 Goal/Trajectory 的运动损失
反向传播得到的隐式任务相关性：如果某些点的语义—几何关系有助于预测终态或轨迹，
relevance head 会提高这些点在 (J) 中的质量；如果场被替换为均匀或打乱分布，context
token 改变并导致运动性能退化。

因此当前最严谨的命名是：

```text
task-conditioned learned Motion Functional Field
```

而不是“从视频直接标注得到的运动场”，也不是未经验证的“真实接触场”。

---

## 6. Goal Pose Diffusion

### 6.1 状态和网络

Goal 状态为一个 9D 终态位姿。平移归一化后使用 DDPM 加噪；旋转 6D 保持连续
表示。每个训练样本随机采样一个扩散时间步，网络预测 clean (x_0)，而不是直接
回归单一均值。

Goal decoder：

```text
noisy Pose9D → 9→128 pose embedding
             → timestep embedding
             → 4 层 GoalConditionBlock
             → 对 3 个 scene tokens cross-attention
             → 9D clean pose prediction
```

每个 block 采用 timestep-conditioned AdaLN、cross-attention、FFN 和 residual。

### 6.2 Goal 损失

\[
\mathcal L_G=
\lambda_d\operatorname{MSE}(\hat x_0,x_0)
 +\lambda_t\operatorname{SmoothL1}(\hat t,t)
 +\lambda_R d_{SO(3)}(\hat R,R).
\]

旋转损失在恢复 (SO(3)) 后计算 geodesic distance，而不是对 6D 分量逐元素做
欧氏误差。这避免了两个不同 6D 数值表示对应同一旋转时的错误惩罚。

### 6.3 多终态采样

推理从不同 Gaussian 初始状态出发，用 DDIM 生成 (K_g) 个终态候选。多样性来自
扩散初始噪声，而不是复制同一个确定性回归结果。当前默认可使用 16 个 goal 候选。

---

## 7. Goal-conditioned Trajectory Diffusion

### 7.1 状态定义

完整标签为 `64×9`。第 0 帧 identity 是已知起点，扩散状态为第 1–63 帧；第 0 帧
在 decoder 内作为 hard start token 参与每层 attention，并在每层更新后重新固定，
因此不会在采样后才突然拼接。

训练时对 GT goal 的 66% 样本加入标准差 0.03 的小扰动，34% 保持 clean。这使轨迹
decoder 学会适应 Goal decoder 的不确定性。

### 7.2 轨迹 decoder

```text
63 个 noisy Pose9D token
 + 离散 sinusoidal frame position
 + hard identity start token
 + embedded goal token
 + 3 个 scene context token
        │
        ▼
6 层 TrajectoryConditionBlock
        ├─ timestep AdaLN
        ├─ temporal Conv1d(kernel=3)
        ├─ non-causal temporal self-attention
        ├─ cross-attention 到 scene+goal memory
        ├─ FFN
        └─ residual connections
        │
        ▼
63 个 clean Pose9D，再恢复 identity 为 64×9
```

与 Goal decoder 相比，Trajectory decoder 增加了明确的时间结构：离散 frame position、
局部 temporal convolution、全序列 self-attention 和 goal-conditioned memory。它
不是把一个 goal 简单复制到所有时间步。

### 7.3 轨迹损失

当前损失包含：

1. 扩散 clean-state reconstruction；第 1 帧权重 20，第 63 帧权重 2；
2. 全部中间帧平移 SmoothL1 和旋转 geodesic；
3. 一阶 translation velocity loss，当前权重 0.5；
4. 二阶 acceleration loss，当前权重 0.1；
5. 第一帧 start-boundary loss，当前权重 2；
6. 末帧 soft endpoint loss，约束轨迹接近 Goal，但不手工覆盖末帧。

这种设计同时约束任务完成、起点连续性和运动平滑性。它也解释了当前的一个实际
现象：低频路径保持较好，但中高频运动细节仍被模型和数据平均化，不能把“平滑”
误写成“完全恢复人类轨迹”。

### 7.4 每个终态生成多条轨迹

给定 (K_g) 个 Goal，每个 Goal 使用独立 trajectory diffusion 初始噪声采样
(K_t) 条轨迹，输出：

```text
goals       [B, K_g, 9]
trajectories[B, K_g, K_t, 64, 9]
```

最终候选可以按照末态误差、碰撞、工作空间、夹爪姿态和任务执行约束进行排序。

---

## 8. 训练基础设施与可复现性

当前训练支持：

- YAML 配置；
- 固定随机种子；
- AdamW；
- AMP（GPU 可用时）；
- 梯度裁剪；
- EMA；
- checkpoint `best.pt` / `last.pt`；
- 断点恢复；
- train/validation 分项损失记录；
- DDPM 训练、DDIM 推理；
- model registry：`three_token_hierarchical_diffusion`；
- 统一接口 `compute_loss(batch)` 和 `sample(batch, ...)`。

本轮 V2 训练配置：

```text
hidden_dim=128
encoder_heads=4
goal_layers=4
trajectory_layers=6
motion_field_temperature=0.25
motion_field_pair_weight=0.25
DDPM train steps=100
DDIM inference steps=20
batch size=16
EMA decay=0.995
```

代码版本、训练产物和数据缓存分离：大 checkpoint、DINO 特征和运行图像不提交到
Git；仓库只保存模型、配置、测试、评估和文档。

---

## 9. 当前实验事实

### 9.1 V1 负向结果

V1 使用两个独立 relevance head，并把各自 softmax 场用于 relation pooling，但仍
保留 raw global token。32 条样本可高度 overfit，任务 loss 很低，但场归一化熵约
0.999、peak mass 约 0.005，接近 256 点均匀分布的 0.003906。说明“加一个 heatmap
head”不足以形成可解释功能场，global bypass 让模型可以绕过场。

### 9.2 V2 结果

V2 通过联合二维关系场和 field bottleneck 消除同容量旁路。全量训练在 epoch 209
早停，最佳 checkpoint 为 epoch 129，验证 total=0.47085。固定 test 的 CPU K=4
评估为：

| 指标 | V0 A3b | V2 Joint Field |
|---|---:|---:|
| Goal top-1 translation | 0.02874 m | 0.02925 m |
| Goal top-1 rotation | 24.42° | 25.72° |
| Trajectory top-1 translation | 0.04290 m | 0.04288 m |
| Trajectory top-1 rotation | 14.00° | 14.04° |

V2 没有明显牺牲原有轨迹任务性能。由于当前评估会话没有可用 GPU，以上完整验证
采用 CPU K=4，而不是 GPU K=16；论文中应把它作为固定诊断结果，不应直接写成大规模
SOTA 对比。

### 9.3 因果消融

在相同测试 episode 和采样协议下：

| field 输入 | Goal 平移 | Trajectory 平移 | Trajectory 旋转 |
|---|---:|---:|---:|
| learned | 0.02932 m | 0.04198 m | 13.77° |
| uniform | 0.03302 m | 0.04357 m | 14.34° |
| rolled | 0.03305 m | 0.04355 m | 14.36° |

均匀或循环打乱场造成退化，证明 Motion Field 参与了预测计算，而不是单纯的
attention visualization。

### 9.4 运动频谱

固定 test 上，GT goal 条件下的位置低/中/高频能量保留率约为：

```text
position: 0.887 / 0.339 / 0.195
velocity: 0.799 / 0.245 / 0.151
```

因此当前方法的真实结论是：低频全局运动和起点连续性较可靠，中高频细节仍然明显
衰减。Motion Field 解决的是条件相关性和功能区域可视化，不等于已经解决 trajectory
high-frequency fidelity。

可视化文件：

```text
/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/fields_visuals/
  train_episode_0.png
  val_episode_15.png
  test_episode_14.png
```

---

## 10. 相对于同领域方法可以突出的点

下面的表述是“可以合理突出”的方法差异，不等同于已经证明的全面性能领先。

### 10.1 相对于 AffCorrs：从区域对应到运动可用的任务场

AffCorrs 的核心是用稠密视觉 descriptor 和循环匹配把源 affordance region 转移到
目标图像；它解决的是“目标中哪个区域对应源查询区域”。当前 LFV 可以突出：

- Stage 1 保留 DINO 稠密语义和正/反向循环匹配；
- 通过 FGW 把完整可见部件内部的结构关系加入 Contact Field transport；
- Stage 2 不把 Contact Field 当作运动网络输入，而是由运动监督学习独立的 Motion
  Functional Field；
- 这个场通过 encoder bottleneck 真正影响 Goal/Trajectory prediction，并经过
  uniform/rolled intervention 进行因果验证。

安全的说法是“从 affordance region transfer 扩展到 task-conditioned motion relevance
field”。不要说成 AffCorrs 原本已经解决了轨迹生成，也不要说当前 V2 的 Motion Field
是 AffCorrs 的直接输出。

### 10.2 相对于 TAX-Pose：相关思想相近，但问题和计算范式不同

TAX-Pose 将任务定义为两个物体之间的 task-specific cross-pose，通过点级软对应和
解析 SVD 估计相对终态姿态，并强调平移等变性和新物体泛化。

LFV 可以突出但必须限定：

- 两者都承认“功能不是物体类别本身，而是对象间任务关系”；
- LFV 的 Motion Field 也提供可视化的点级任务相关性；
- 但 LFV 不是 SVD 解析 cross-pose，而是条件 Goal Diffusion + Trajectory Diffusion；
- LFV 生成多个不确定 Goal 和每个 Goal 的多条完整 64 帧路径；
- LFV 的场是由运动损失学习的软关系瓶颈，不是 TAX-Pose 的同一 correspondence
  estimator，也没有当前版本的严格 translation-equivariant 证明。

因此建议使用“受 task-specific correspondence 启发的运动相关性建模”，不要声称
“实现了 TAX-Pose 式等变泛化”。

### 10.3 相对于 Diffusion Policy：结构化的刚体目标与轨迹分解

Diffusion Policy 将动作序列作为条件扩散输出，并使用时间序列 Transformer 进行
视觉条件动作生成。LFV 的可突出点是：

- 先生成具有物理含义的 object-centric 9D Goal Pose；
- 再用 Goal-conditioned trajectory diffusion 生成 64 帧物体轨迹；
- Goal 与 trajectory 的不确定性被显式分层采样；
- 轨迹在 object/camera/world 坐标链中有明确几何解释；
- 最后可接入抓取候选、固定 attachment、IK 和碰撞筛选。

这使系统更适合需要终态约束和后验可行性筛选的长时程操作。不要把它描述为比
Diffusion Policy 普遍更强；当前只证明了本任务和本数据协议下的可运行性。

### 10.4 相对于 3D point-cloud diffusion 方法：语义—几何同步与双对象关系

3D Diffusion Policy、3D Diffuser Actor 等方法说明点云可以作为扩散策略的条件输入。
LFV 的差异在于：

- DINO 特征和 XYZ 使用完全相同的采样索引；
- manipulated/reference 两个对象采用角色不对称的 PointNet 和双向 cross-attention；
- 不是把整场景压缩为单一全局向量，而是保留 initial、manipulated-query-reference、
  reference-query-manipulated 三类 context；
- Motion Field 使“哪些点对任务运动有用”成为可视化且可干预的网络状态。

---

## 11. 论文中应主动回避或谨慎表述的内容

### 11.1 不要声称严格跨实例泛化

当前数据缺少可靠 `object_instance_id`，split 只保证 episode 不泄漏。因此不能写：

```text
strict object-instance-disjoint generalization
unseen-object generalization has been demonstrated
```

可以写：

```text
episode-disjoint baseline on the current pouring dataset
```

严格跨实例实验需要补充实例 ID、按实例划分 train/test，并重新训练与测试。

### 11.2 不要把 Motion Field 写成直接视频监督

当前 Motion Field 没有逐点 GT。正确因果关系是：

```text
Goal/Trajectory labels → diffusion loss → relevance head → Motion Field
```

它是 learned task relevance field，不是从视频直接读取的 ground-truth affordance。

### 11.3 不要声称解决单视角完整几何问题

Stage 1 热力迁移首先作用于可观测区域；仿真中的完整点云补全/热力扩散是执行层的
几何支撑。真实单视角 RGB-D 下是否能获得另一侧几何，需要单独的离线重建、CAD、
多视角或 SAM3D 实验，不能从当前 Stage 2 结果推出。

### 11.4 不要声称 Motion Field 已经恢复人类高频动作

当前频谱结果显示中高频保留率较低，轨迹有平滑化倾向。应突出“低频任务运动、
终态和起点连续性”，并把高频轨迹恢复作为限制和后续工作。

### 11.5 不要把可视化峰值直接等同于真实接触点

Motion Field 是任务运动相关性；Contact Field 才是 Stage 1 的接触热力。两者都可能
出现局部峰值，但物理含义不同。论文图注必须标明是 `contact field` 还是
`motion functional field`。

### 11.6 不要把仿真执行结果混写成网络精度

夹爪几何、attachment、碰撞、IK、控制时序和相机坐标变换都会影响执行。网络误差、
候选筛选失败和控制失败应分开统计。

---

## 12. 建议的论文叙事重点

推荐把贡献凝练成以下逻辑，而不是罗列大量模块：

1. **Task knowledge factorization**：把示范中的接触语义和运动语义分开建模；
2. **Semantic–structural contact transfer**：用 DINO 循环对应和 FGW 将源 Contact
   Field 迁移到目标功能部件；
3. **Learned motion functional field**：在双对象 XYZ–DINO encoder 中学习一个受
   Goal/Trajectory 监督的联合关系场，并以 field bottleneck 连接到两个扩散分支；
4. **Hierarchical SE(3) generation**：先采样终态，再生成多条 goal-conditioned
   64-step 轨迹；
5. **Executable realization**：将 Contact-guided grasp、object-to-TCP attachment
   和几何可行性筛选接到生成结果之后。

最有说服力的消融顺序是：

```text
V0：无显式 Motion Field
V1：独立 relevance fields，但保留 global bypass
V2：joint pair field + field bottleneck
V2 intervention：uniform / rolled field
```

这条证据链比单独展示一张漂亮热力图更重要，因为它回答了“场是否真实参与计算”。

---

## 13. 当前代码、模型和复现实验入口

代码仓库：

```text
/home/users1/ljian/LFV_stage2_motion_field
```

模型核心文件：

```text
lfv/models/functional_motion_generation/encoders/bidirectional_scene_encoder.py
lfv/models/functional_motion_generation/system.py
lfv/models/functional_motion_generation/goal/decoder.py
lfv/models/functional_motion_generation/goal/diffuser.py
lfv/models/functional_motion_generation/trajectory/decoder.py
lfv/models/functional_motion_generation/trajectory/diffuser.py
```

训练配置：

```text
configs/stage2/motion_field_v2_pouring_lfv.yaml
```

测试：

```bash
python -m pytest -q tests/stage2
```

测试结果：`33 passed`。

最佳 checkpoint：

```text
/home/users1/ljian/lfv_runs/stage2/motion_functional_field/v2_joint/checkpoints/best.pt
```

Git 版本：

```text
branch: stage2/motion-functional-field
commit: 68c0f2a
tag:    stage2-motion-field-v2.2-trained
```

---

## 14. 相关工作定位

- AffCorrs：One-Shot Transfer of Affordance Regions，解决视觉 affordance 区域的
  one-shot 对应；LFV 在 Stage 1 复用其语义循环对应思想，并增加连续热力和结构
  transport。
- TAX-Pose：Task-Specific Cross-Pose Estimation，学习任务相关的跨对象姿态关系；
  LFV 与其共享“功能关系优先”的思想，但使用显式 Motion Field 和分层扩散生成，
  不等同于 TAX-Pose 的软对应 + SVD。
- Diffusion Policy：将时序动作作为条件扩散输出；LFV 将刚体 Goal Pose 与完整
  trajectory 解耦，并保留 object-centric 坐标和候选层级。
- 3D Diffuser Actor / 3D Diffusion Policy：证明点云条件扩散适用于机器人操作；
  LFV 强调逐点 XYZ–DINO 对齐、双对象角色关系和可干预运动场。

建议引用的原始资料：

1. [AffCorrs, PMLR 205](https://proceedings.mlr.press/v205/hadjivelichkov23a.html)
2. [TAX-Pose, PMLR 205](https://proceedings.mlr.press/v205/pan23a.html)
3. [Diffusion Policy, arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
4. [3D Diffusion Policy, arXiv:2403.03954](https://arxiv.org/abs/2403.03954)
5. [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)

---

## 15. 一句话版本

LFV 是一个以同步 XYZ–DINO 双对象表示为基础、先迁移可执行接触场、再通过联合
Motion Functional Field 约束 Goal Pose 和 64 步 SE(3) 轨迹扩散，并最终接入抓取与
几何可行性筛选的模块化机器人操作系统；当前最可信的贡献是“运动相关性场成为扩散
生成的真实中间状态并可被因果干预验证”，而不是已经完成严格跨实例泛化或高频轨迹
复现。
