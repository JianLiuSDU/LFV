# LFV 完整方法材料：跨实例任务场迁移与功能关系运动生成

**文档用途**：为论文写作、实验设计和后续实现提供统一的方法材料。本文档不是论文
Method 章节的最终排版稿，而是对 LFV 方法的完整技术描述，包含方法动机、计算流程、
网络结构、训练目标、推理方式、核心贡献和表述边界。

**方法版本**：按当前确定的理想方法叙事编写。文中将“已经定义的方法目标”和“当前
代码可能尚未完全验证的实现细节”分开，不把未完成实验写成实验结论。

---

## 1. 方法总览

LFV 的目标是从人类演示中获得可解释、可迁移的任务知识，并将其用于同类别新实例
上的抓取、目标状态预测和运动轨迹生成。方法将任务知识分解成两类不同语义的场：

1. **Contact Field**：描述物体表面哪些位置适合与机器人或手部发生接触，主要服务
   于抓取候选生成和抓取实例化；
2. **Motion Functional Field**：描述物体表面哪些位置对当前物体间任务运动最重要，
   主要服务于 Goal Pose 和 Trajectory 的生成。

两种场的职责不同。Contact Field 可以由人类接触证据直接构造并通过 Stage 1 迁移；
Motion Functional Field 默认不是数据集中的人工标签，而是在 Stage 2 的对象关系网络
中由 relevance head 预测，并通过目标位姿和轨迹扩散损失学习。

完整方法的因果链为：

```text
人类演示
  ├─ 手—物接触证据 → 源 Contact Field A_s
  ├─ 对象三维运动 → Goal T_g 和轨迹 τ
  └─ XYZ–DINO 场景表示
          │
          ├─ Stage 1：Contact Field 跨实例迁移
          │       Soft Heatmap AffCorrs + FGW
          │       → 目标实例 Contact Field A_t
          │
          └─ Stage 2：功能关系运动生成
                  XYZ–DINO Object Encoder
                  → 双向 Cross-Attention
                  → Motion Functional Field R
                  → 三个关系 Context Tokens C
                  → Goal Pose Diffusion
                  → Goal-Conditioned Trajectory Diffusion
                          │
                          └─ Stage 3：抓取与机器人执行
                                  Contact 引导抓取
                                  IK/碰撞检查/TCP 执行
```

方法的核心不是简单地堆叠 DINO、PointNet、Transformer 和 Diffusion，而是建立如下
明确关系：

```text
语义对应：目标中哪个区域是任务相关功能部件
结构对应：源场在目标功能部件内部应该落在哪里
功能编码：哪些局部区域真正决定物体运动
分层生成：先确定任务完成状态，再生成到达该状态的完整轨迹
```

---

## 2. 问题定义和坐标约定

### 2.1 双对象任务

每个操作任务由一个被操作物体 `m` 和一个参考物体 `r` 构成，例如：

- 倒水：杯子是被操作物体，碗或接收容器是参考物体；
- 拉抽屉：抽屉是被操作物体，柜体或桌面是参考物体；
- 放置任务：被放置物体是 `m`，目标支撑物是 `r`。

Stage 2 的输入为两组点云和对应的 DINO 稠密特征：

\[
X^m=\{x_i^m\}_{i=1}^{N_m},\qquad
X^r=\{x_j^r\}_{j=1}^{N_r},
\]

\[
D^m=\{d_i^m\}_{i=1}^{N_m},\qquad
D^r=\{d_j^r\}_{j=1}^{N_r}.
\]

其中 (x\in\mathbb R^3) 是物体中心坐标系或统一相机坐标系下的点位置，(d) 是
冻结 DINOv2 提取的语义描述符。

### 2.2 Goal 和 Trajectory

终态目标使用对象中心的 9D 位姿表示：

\[
g=[t_x,t_y,t_z,r_1,r_2,r_3,r_4,r_5,r_6],
\]

前三维是平移，后六维是连续 6D 旋转表示。6D 表示通过正交化转换为旋转矩阵，
避免欧拉角不连续和四元数符号歧义。

对象运动轨迹表示为：

\[
\tau=\{g^{(0)},g^{(1)},\ldots,g^{(H-1)}\},
\qquad g^{(h)}\in\mathbb R^9.
\]

轨迹可以采用初始对象坐标系中的绝对位姿，也可以采用相对于第一帧的残差位姿。训练、
归一化、推理和执行必须使用同一种定义，并明确第一帧的锚定方式。

---

## 3. Stage 1：Contact Field 的构造和跨实例迁移

### 3.1 源 Contact Field 的构造

从人类演示中选取手部与物体发生有效接触的帧，利用手部关键点、手部遮挡区域或
手—物距离证据，在物体表面上构造连续接触概率：

\[
A_s(u,v)\in[0,1]
\]

或者在点云上表示为：

\[
h_i^s\in[0,1].
\]

该场可以具有局部峰值，例如杯把手中央或抽屉把手的稳定抓取区域。连续表示比二值
mask 保留更多信息，后续可以区分高热中心与低热边缘。

Contact Field 的来源是手—物接触证据；它与 Motion Functional Field 不同，不能把
二者写成同一个场。

### 3.2 Soft Heatmap AffCorrs：语义功能部件定位

源输入为：

\[
I_s,M_s,A_s,
\]

目标输入为：

\[
I_t,M_t.
\]

源和目标图像采用相同的 mask 包围框、留边、等比例缩放和 letterbox 预处理，并保存
原图坐标与 DINO patch 网格之间的映射。

冻结 DINOv2 提取稠密 patch descriptor，并逐向量 L2 归一化：

\[
f\leftarrow\frac{f}{\lVert f\rVert_2+\epsilon}.
\]

源和目标 mask 内特征分别记为：

\[
F_s=\{f_i^s\}_{i=1}^{N_s},qquad
F_t=\{f_j^t\}_{j=1}^{N_t}.
\]

原版 AffCorrs 的核心来自 *One-Shot Transfer of Affordance Regions? AffCorrs!*：

- 使用 DINO-ViT 稠密特征；
- 将源查询区域聚类为 prototypes；
- 将目标前景过分割为候选区域；
- 源区域向目标区域正向匹配；
- 目标区域反向匹配源图像全部前景；
- 正向支持度和反向落回查询区域概率相乘。

LFV 将原版二值 query 改为连续热力加权。源正热集合为：

\[
\mathcal P_s=\{i\mid a_i^s>\tau_{pos}\}.
\]

对正热 patch 进行加权 K-Means：

\[
\min_{z_k^s,c_i}
\sum_{i\in\mathcal P_s}a_i^s
\lVert f_i^s-z_{c_i}^s\rVert_2^2.
\tag{1}
\]

第 (k) 个源 prototype 的热力权重为：

\[
m_k=\sum_{i:c_i=k}a_i^s,qquad
\omega_k=\frac{m_k}{\sum_r m_r+\epsilon}.
\tag{2}
\]

目标功能部件内的 patch 做较密集的 K-Means：

\[
\min_{z_j^t,d_l}
\sum_l\lVert f_l^t-z_{d_l}^t\rVert_2^2.
\tag{3}
\]

源 prototype 和目标区域 prototype 的相似度为：

\[
S_{kj}=\langle z_k^s,z_j^t\rangle.
\tag{4}
\]

沿目标区域进行正向 Softmax：

\[
P_{kj}^{f}
=
\frac{\exp(S_{kj}/\tau_f)}
{\sum_r\exp(S_{kr}/\tau_f)}.
\tag{5}
\]

热力加权正向投票为：

\[
V_j=\sum_k\omega_kP_{kj}^{f}.
\tag{6}
\]

目标区域向源全部前景 patch 反向匹配：

\[
B_{ji}=\langle z_j^t,f_i^s\rangle,qquad
P_{ji}^{b}
=
\frac{\exp(B_{ji}/\tau_b)}
{\sum_r\exp(B_{jr}/\tau_b)}.
\tag{7}
\]

源连续热力归一化为：

\[
\bar A_s(i)=\frac{a_i^s}{\sum_r a_r^s+\epsilon}.
\tag{8}
\]

反向落回 Contact 区域的概率为：

\[
Q_j=\sum_iP_{ji}^{b}\bar A_s(i).
\tag{9}
\]

最终语义区域分数为：

\[
H_j^{aff}=V_jQ_j.
\tag{10}
\]

将区域分数插值回目标图像并乘以目标 mask，得到：

\[
A_{aff}(u,v)=M_t(u,v)\cdot
\operatorname{Norm}\left[\operatorname{Interp}^{-1}(H_{d(u,v)}^{aff})\right].
\tag{11}
\]

这里的 (A_{aff}) 只负责判断目标中哪个区域与源功能部件语义对应，不能保证源
Contact 峰值在目标部件内部的位置仍然正确。

### 3.3 RGB-D 提升和部件内部结构

对源、目标功能部件的有效深度像素反投影：

\[
p(u,v)=D(u,v)K^{-1}[u,v,1]^\top
=
\left(
\frac{(u-c_x)z}{f_x},
\frac{(v-c_y)z}{f_y},z
\right)^\top.
\tag{12}
\]

源和目标得到完整可见功能部件点云：

\[
P_s=\{p_i^s\},qquad P_t=\{p_j^t\}.
\]

FGW 的输入必须是整个源和目标功能部件，而不是只输入高热 Contact 点。因为高热
区域的结构位置只有相对于整个部件才能被确定。

为了控制计算量，对两个点云进行 FPS 或 voxel 下采样到约 256–512 个节点，并保存
原始点与下采样点之间的映射。

在下采样点云上分别建立 kNN 图，边权为欧氏距离：

\[
w_{ik}^s=\lVert p_i^s-p_k^s\rVert_2,qquad
w_{jl}^t=\lVert p_j^t-p_l^t\rVert_2.
\tag{13}
\]

通过图最短路径得到部件内部 geodesic 距离：

\[
D_s(i,k)=\operatorname{ShortestPath}_{G_s}(i,k),qquad
D_t(j,l)=\operatorname{ShortestPath}_{G_t}(j,l).
\tag{14}
\]

再使用非零距离的 median、q95 或 diameter 归一化：

\[
\widetilde D_s=\operatorname{clip}\left(\frac{D_s}{q_s+\epsilon},0,d_{max}\right),
\qquad
\widetilde D_t=\operatorname{clip}\left(\frac{D_t}{q_t+\epsilon},0,d_{max}\right).
\tag{15}
\]

这样结构项表达的是部件内部的相对距离，而不是相机坐标、绝对尺度或人工定义的
左右方向。

### 3.4 FGW 结构传输

源、目标点的 DINO 语义代价为：

\[
C_{ij}^{sem}=1-\langle f_i^s,f_j^t\rangle.
\tag{16}
\]

对源、目标节点使用均匀质量：

\[
a_i=1/N_s,qquad b_j=1/N_t.
\tag{17}
\]

若 (T_{ij}) 表示源点 (i) 到目标点 (j) 的传输质量，则：

\[
\Pi(a,b)=
\{T\ge0\mid T\mathbf1=a,\;T^\top\mathbf1=b\}.
\tag{18}
\]

标准 FGW 目标为：

\[
\min_{T\in\Pi(a,b)}
(1-\alpha)\sum_{i,j}C_{ij}^{sem}T_{ij}
+\alpha\sum_{i,k,j,l}
\left(\widetilde D_s(i,k)-\widetilde D_t(j,l)\right)^2
T_{ij}T_{kl}.
\tag{19}
\]

其中 (alpha) 控制语义和结构的平衡。建议使用 (alpha=0.5)，并进行
({0.2,0.5,0.8}) 消融。

FGW 求得软传输矩阵后，不重新执行目标部件分割，而是直接输运源 Contact Field：

\[
h_j^{fgw}
=
\frac{\sum_iT_{ij}h_i^s}
{\sum_iT_{ij}+\epsilon}.
\tag{20}
\]

再通过 3NN 或局部插值恢复到目标原始部件点云：

\[
h_q=
\frac{\sum_{r\in\mathcal N_3(q)}
\frac{h_r^{fgw}}{\lVert p_q-p_r\rVert_2+\epsilon}}
{\sum_{r\in\mathcal N_3(q)}
\frac1{\lVert p_q-p_r\rVert_2+\epsilon}}.
\tag{21}
\]

最终将点云热力恢复到二维图像并与语义场融合：

\[
A_t^{contact}(u,v)=
M_t(u,v)A_{fgw}(u,v)
\left[\lambda+(1-\lambda)A_{aff}(u,v)^\gamma\right].
\tag{22}
\]

因此：

```text
AffCorrs：目标中哪个功能部件正确
FGW：该功能部件内部哪个位置对应源 Contact 峰值
式(20)：源连续 Contact Field 如何被直接输运
```

### 3.5 Stage 1 的输入输出边界

Stage 1 输出目标实例可见功能部件上的 Contact Field：

\[
A_t^{contact}(u,v)\in[0,1].
\]

同时输出目标点云及点级热力：

\[
P_t=\{p_j^t\},qquad h_t=\{h_j^t\}.
\]

该结果可以被完整点云补全、GraspNet、碰撞检查和抓取候选筛选使用。但 Stage 1 本身：

- 不负责预测 Goal；
- 不负责生成中间轨迹；
- 不自动补全单视角不可见表面；
- 不将 Contact Field 当作 Stage 2 的初始运动观测。

---

## 4. Stage 2：Motion Functional Field 与分层扩散生成

### 4.1 为什么需要 Motion Functional Field

Contact Field 只说明哪里适合接触，而完整任务运动还需要回答：

- 哪些物体区域决定目标状态；
- 哪些区域与参考物体形成关键关系；
- 哪些局部几何和语义信息应被 Goal Decoder 重点读取；
- 哪些局部关系应被 Trajectory Decoder 保留。

因此，Stage 2 定义并学习一个任务相关的表面场：

\[
R^a=\{r_i^a\}_{i=1}^{N_a},qquad a\in\{m,r\},
\]

其中 (r_i^ain[0,1]) 表示对象表面点 (i) 对当前任务运动的相关程度。

该场与 Stage 1 的 Contact Field 语义不同：

| 场 | 含义 | 主要作用 |
|---|---|---|
| Contact Field | 适合发生手/夹爪接触的位置 | 抓取候选生成 |
| Motion Functional Field | 对任务运动和对象间关系重要的位置 | Goal/Trajectory 生成 |

### 4.2 双对象 XYZ–DINO 编码

被操作物体和参考物体分别编码：

\[
u_i^m=E_m([x_i^m,d_i^m]),qquad
u_j^r=E_r([x_j^r,d_j^r]).
\tag{23}
\]

两个编码器可以采用结构相同但参数独立的 PointNet 或 PointNet++。这种设计保留了
两个物体的角色差异，并避免把所有局部点过早压成一个无角色的全局向量。

### 4.3 双向 Cross-Attention

从被操作物体看参考物体：

\[
\widetilde u_i^m
=
\operatorname{CrossAttn}
(u_i^m,U^r,U^r).
\tag{24}
\]

从参考物体看被操作物体：

\[
\widetilde u_j^r
=
\operatorname{CrossAttn}
(u_j^r,U^m,U^m).
\tag{25}
\]

双向关系特征分别描述两个方向上的任务相关性，而不是只使用一个对称的全局场景
embedding。

### 4.4 Relevance Head 产生 Motion Functional Field

对每个方向，显式 relevance head 预测逐点运动相关性：

\[
r_i^a=
\sigma\left(
h_a([u_i^a,\widetilde u_i^a])
\right),
\quad (a,b)\in\{(m,r),(r,m)\}.
\tag{26}
\]

这里的 (r_i^a) 是网络在线产生的 Motion Functional Field，不是预先给定的输入，
也不是 Stage 1 FGW 的输出。

如果没有独立场标签，它通过 Goal 和 Trajectory 的去噪损失间接学习；如果具有可靠
的伪场或一致性目标，可以额外加入：

\[
\mathcal L_{field}.
\]

要在论文中把它称为具有明确语义的功能场，至少应满足：

1. relevance head 显式输出点级场值；
2. 场值能够被可视化；
3. 置零、打乱或移除该场会影响 Goal/Trajectory 预测；
4. 场值在不同轨迹或视角下具有稳定的功能区域响应。

### 4.5 功能场加权的三个 Context Tokens

利用 Motion Functional Field 加权聚合关系特征：

\[
c_{a\leftarrow b}
=
\frac{\sum_i r_i^a\widetilde u_i^a}
{\sum_i r_i^a+\epsilon}.
\tag{27}
\]

同时对两个对象的整体特征进行池化：

\[
c_0=\operatorname{Pool}(U^m\cup U^r).
\tag{28}
\]

最终关系上下文为：

\[
C=[c_0,c_{m\leftarrow r},c_{r\leftarrow m}]
\in\mathbb R^{B\times3\times C}.
\tag{29}
\]

三个 token 的含义为：

- (c_0)：双对象的整体语义和几何上下文；
- (c_{m\leftarrow r})：从被操作物体角度看参考物体的任务相关关系；
- (c_{r\leftarrow m})：从参考物体角度看被操作物体的任务相关关系。

这是 Stage 2 区别于普通点云扩散策略的关键：模型不是对所有点做均匀池化，而是先
预测功能场，再使用该场选择真正与任务运动相关的局部关系。

### 4.6 Goal Pose Diffusion

将归一化目标位姿 (g_0) 加噪：

\[
g_t
=
\sqrt{\bar\alpha_t}g_0
+\sqrt{1-\bar\alpha_t}\epsilon_g,
\qquad
\epsilon_g\sim\mathcal N(0,I).
\tag{30}
\]

目标位姿 token 为：

\[
q_g=\operatorname{MLP}_g(g_t)+e_t,
\tag{31}
\]

其中 (e_t) 是 diffusion timestep embedding。

Goal Decoder 由若干标准 Transformer block 构成，每个 block 包含：

1. timestep-conditioned AdaLN 或 FiLM；
2. Goal token self-attention；
3. Goal-to-scene cross-attention；
4. residual FFN。

其条件交互可以表示为：

\[
q_g^{(l+1)}
=q_g^{(l)}+\operatorname{SelfAttn}(q_g^{(l)}),
\]

\[
q_g^{(l+1)}
\leftarrow q_g^{(l+1)}
+\operatorname{CrossAttn}(q_g^{(l+1)},C,C).
\tag{32}
\]

最终噪声预测为：

\[
\widehat\epsilon_g=H_g(q_g^{(L)}).
\tag{33}
\]

目标位姿损失为：

\[
\mathcal L_{goal}
=
\lVert\epsilon_g-\widehat\epsilon_g\rVert_2^2.
\tag{34}
\]

Motion Functional Field 影响 Goal 的路径为：

\[
R\rightarrow C\rightarrow
\operatorname{CrossAttn}_{goal}
\rightarrow\widehat\epsilon_g
\rightarrow\widehat T_g.
\]

### 4.7 Goal-Conditioned Trajectory Diffusion

将长度为 (H) 的对象轨迹整体加噪：

\[
\tau_t
=
\sqrt{\bar\alpha_t}\tau_0
+\sqrt{1-\bar\alpha_t}\epsilon_\tau.
\tag{35}
\]

第 (h) 帧轨迹 token 为：

\[
q_h
=
\operatorname{MLP}_{traj}(\tau_t^{(h)})+p_h+e_t,
\tag{36}
\]

其中 (p_h) 是时间位置或 phase embedding，(e_t) 是扩散时间嵌入。

Trajectory Decoder 的核心结构为：

1. 时间条件 AdaLN 或 FiLM；
2. 非因果时间 self-attention；
3. trajectory-to-scene cross-attention；
4. trajectory-to-goal cross-attention；
5. residual FFN；
6. 逐时间步噪声预测头。

形式上：

\[
Q_\tau^{(l+1)}
=Q_\tau^{(l)}+\operatorname{SelfAttn}(Q_\tau^{(l)}),
\]

\[
Q_\tau^{(l+1)}
\leftarrow Q_\tau^{(l+1)}
+\operatorname{CrossAttn}(Q_\tau^{(l+1)},C,C),
\]

\[
Q_\tau^{(l+1)}
\leftarrow Q_\tau^{(l+1)}
+\operatorname{CrossAttn}(Q_\tau^{(l+1)},q_g,q_g).
\tag{37}
\]

最终输出：

\[
\widehat\epsilon_\tau=H_\tau(Q_\tau^{(L)}).
\tag{38}
\]

轨迹损失为：

\[
\mathcal L_{traj}
=
\frac1H\sum_{h=1}^{H}
\lVert\epsilon_\tau^{(h)}-widehat\epsilon_\tau^{(h)}\rVert_2^2.
\tag{39}
\]

Motion Functional Field 对轨迹有两条作用路径：

\[
R\rightarrow C\rightarrow\operatorname{TrajectoryDecoder},
\]

以及：

\[
R\rightarrow C\rightarrow\operatorname{GoalDecoder}
\rightarrow\widehat T_g
\rightarrow\operatorname{TrajectoryDecoder}.
\]

因此轨迹不是简单地对目标位姿做线性插值，而是在功能关系上下文、目标状态和时间
序列结构共同约束下生成。

### 4.8 联合优化和推理

两个扩散分支共享场景 Encoder，但拥有独立的 Decoder：

\[
\mathcal L
=
\lambda_g\mathcal L_{goal}
+\lambda_\tau\mathcal L_{traj}
+\lambda_f\mathcal L_{field}.
\tag{40}
\]

轨迹损失必须沿时间维度取平均，避免由于轨迹长度导致其数值天然大于 Goal 损失。

推理时先计算一次上下文：

\[
C=\operatorname{Encoder}(X^m,D^m,X^r,D^r).
\]

然后从不同高斯噪声初始化多个 Goal：

\[
g_T^{(k)}\sim\mathcal N(0,I),qquad k=1,ldots,K_g.
\]

对每个 Goal 再独立采样多条轨迹：

\[
\tau_T^{(k,l)}\sim\mathcal N(0,I),qquad l=1,ldots,K_\tau.
\]

最终输出 (K_g) 组目标位姿，每个目标位姿对应 (K_\tau) 条轨迹。不同样本来自
不同扩散噪声，而不是将一个确定性预测复制多次。

---

## 5. Stage 3：从对象运动到机器人执行

前三个模块都在对象空间中运行。执行阶段才引入机器人本体：

1. 使用目标 Contact Field 在完整点云上筛选抓取候选；
2. 生成满足碰撞和夹爪几何约束的抓取位姿；
3. 通过固定的 object-to-gripper attachment 将对象轨迹转换为 TCP 轨迹；
4. 使用 IK、工作空间和碰撞检查过滤候选；
5. 选择可执行的 Goal–Trajectory 组合；
6. 控制机械臂完成抓取和后续任务。

完整点云补全、SAM3D、GraspNet、夹爪延长件、机器人控制器和 Aubo/Franka 适配属于
执行后端，不应被写成 Stage 1 迁移算子或 Stage 2 运动网络的核心组成。

---

## 6. 方法真正应该突出的重点

### 6.1 连续 Contact Field，而不是二值 affordance mask

连续场保留了功能部件内部的强弱分布。相比只输出“把手区域”的二值结果，连续场
可以表达把手中央高、两端低等对抓取有意义的空间差异。

### 6.2 语义定位和结构定位解耦

Soft Heatmap AffCorrs 负责语义部件定位，FGW 负责部件内部结构对应。这样可以解释
为什么目标部件整体语义正确，但热力峰仍然可能错误，以及 FGW 如何修复这个问题。

### 6.3 Motion Functional Field 是显式中间表示

Stage 2 不仅输出一个 Goal 和一条轨迹，还显式输出逐点的运动相关场。该场可以被
可视化、打乱、置零和消融，从而分析模型是否真正关注了杯口、把手或抽屉等功能区域。

### 6.4 功能场同时影响 Goal 和 Trajectory

Motion Functional Field 先决定场景关系 token，再同时影响目标位姿和中间轨迹。Goal
和 Trajectory 不是两个互相独立的预测器，而是：

\[
p(T_g,\tau\mid C)
=p_{\theta_g}(T_g\mid C)
p_{\theta_\tau}(\tau\mid T_g,C).
\]

### 6.5 分层生成而不是直接回归整段动作

先生成终态，再在目标条件下生成完整轨迹，可以将“任务完成状态”和“到达该状态的
运动过程”分开建模，同时保留二者之间的条件关系。

---

## 7. 论文中的证据边界

论文中应明确以下事实：

1. DINOv2 提供通用稠密语义特征，不直接监督 Contact 或 Motion Field；
2. AffCorrs 原论文使用二值 query 和循环区域对应，连续热力加权是 LFV 的改造；
3. FGW 提供语义—结构联合传输优化，直接输运 Contact Field 是 LFV 的任务化应用；
4. Motion Functional Field 默认由 Stage 2 relevance head 预测，不是 FGW 输出；
5. Contact Field 和 Motion Functional Field 不能混称；
6. 单视角目标观测只提供当前可见表面，隐藏侧补全属于执行后端；
7. 如果训练集只包含同一个物体实例的多条轨迹，只能证明视角或轨迹泛化，不能严格
   声称跨实例泛化；
8. 如果没有显式场监督，应写“learned through task-level denoising objectives”，
   不应写成“directly annotated from demonstrations”。

---

## 8. 建议的实验组织

### 8.1 Contact Field 迁移实验

比较：

- AffCorrs only；
- 最近邻语义匹配；
- 无 FGW；
- 无连续热力加权；
- 仅高热区域输入 FGW；
- 不同 (alpha)；
- 不同目标聚类数。

指标：Contact MSE、Soft-IoU/AUPRC、峰值位置误差、部件内外能量比、热力熵、下游
抓取成功率。

### 8.2 Motion Functional Field 实验

至少比较：

- 完整方法；
- uniform pooling；
- Motion Field 置零；
- Motion Field 随机打乱；
- 移除 relevance head；
- 移除双向 cross-attention。

同时保存：

- 被操作物体上的场；
- 参考物体上的场；
- 双向 cross-attention；
- 三个 context token 的响应；
- Goal 和 Trajectory 的误差变化。

### 8.3 分层扩散实验

比较：

- 直接 Goal 回归；
- Goal Diffusion；
- 无 Goal 条件的 Trajectory Diffusion；
- Goal-conditioned Trajectory Diffusion；
- 直接生成轨迹；
- 仅使用全局 token；
- 使用三个功能关系 token。

指标：终态平移误差、旋转测地误差、轨迹位置误差、终点一致性、轨迹平滑度、
Best-of-K 和任务成功率。

### 8.4 泛化实验

应按照 `object_instance_id` 划分训练、验证和测试集，禁止同一物体实例泄漏。可以
进一步分别测试：

- 新实例；
- 新视角；
- 新初始相对位姿；
- 新轨迹执行速度；
- 新场景布局。

如果只有一个实例的多条视频，应将实验结论限定为视角、姿态和轨迹分布建模，不应
把它包装为跨实例泛化。

---

## 9. 参考工作与 LFV 的关系

- **DINOv2**：提供冻结的稠密视觉语义描述符。
  [DINOv2](https://arxiv.org/abs/2304.07193)
- **AffCorrs**：提供 DINO 描述符、query prototype、正向匹配、反向循环验证的
  语义区域对应骨架。
  [One-Shot Transfer of Affordance Regions? AffCorrs!](https://arxiv.org/abs/2209.07147)
- **FGW**：提供同时融合节点特征和内部结构的软传输目标。
  [Fused Gromov-Wasserstein Distance for Structured Objects](https://arxiv.org/abs/1811.02834)
- **2D-to-3D affordance grounding**：支持将二维交互/ affordance 通过深度提升为
  三维点云表示。
  [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Grounding_3D_Object_Affordance_from_2D_Interactions_in_Images_ICCV2023_paper.pdf)
- **Diffusion Policy**：提供条件动作扩散、噪声预测和时间序列 Transformer 的
  基础形式。
  [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- **DP3**：说明轻量 3D 点云表示可以作为扩散策略的条件。
  [3D Diffusion Policy](https://arxiv.org/abs/2403.03954)
- **3D Diffuser Actor**：说明 token 化 3D 场景可以与去噪 Transformer 联合生成
  位姿轨迹。
  [3D Diffuser Actor](https://arxiv.org/abs/2402.10885)
- **TAX-Pose**：提供显式逐点对应和 importance weight 的可解释关系建模动机，但
  LFV 不直接复制其 cross-pose SVD 求解器。
  [TAX-Pose](https://arxiv.org/abs/2211.09325)

---

## 10. 当前文档入口

为避免历史实验记录和重复设计造成歧义，当前只维护以下入口：

- [`docs/README_zh.md`](../README_zh.md)：文档索引与阅读顺序；
- [`docs/project_architecture_and_development_guide_zh.md`](../project_architecture_and_development_guide_zh.md)：代码结构与开发约束；
- [`docs/stage1_affcorrs_fgw_contact_transfer_zh.md`](../stage1_affcorrs_fgw_contact_transfer_zh.md)：Stage 1 接触迁移；
- [`docs/stage2/current_method_complete_zh.md`](../stage2/current_method_complete_zh.md)：Stage 2 运动生成；
- [`docs/deployment/strict_camera_inference_zh.md`](../deployment/strict_camera_inference_zh.md)：RGB-D 推理入口；
- [`docs/deployment/aubo_camera_execution_bundle_zh.md`](../deployment/aubo_camera_execution_bundle_zh.md)：机器人执行交付。

---

## 11. 最终方法摘要

LFV 首先从人类演示中构造连续 Contact Field，并采用 Soft Heatmap AffCorrs 定位目标
实例中的语义功能部件，再使用 FGW 在完整可见部件内部建立语义—结构对应，将源
Contact Field 直接输运到目标点云和图像。随后，Stage 2 以双对象 XYZ–DINO 为输入，
通过独立对象编码器和双向 cross-attention 建立物体间关系，由 relevance head 显式
预测 Motion Functional Field，并利用该场构造三个关系上下文 token。Goal Pose Diffusion
首先根据这些 token 生成任务完成状态，Trajectory Diffusion 再同时读取场景关系和
Goal 条件，生成实现该状态的完整对象轨迹。最后，Contact Field、目标位姿和对象轨迹
被执行后端转换为机器人抓取与 TCP 运动。

因此，LFV 的核心贡献可以概括为：

> 用连续任务场表示演示知识，用语义—结构对应实现 Contact Field 的跨实例迁移，用
> 显式 Motion Functional Field 选择任务相关关系，并通过 Goal–Trajectory 分层扩散
> 在对象中心空间中生成可执行运动。
