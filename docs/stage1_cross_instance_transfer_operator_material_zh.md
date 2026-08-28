# Stage 1 跨实例任务场迁移算子：论文写作材料与计算细节

**版本**：理想方法定义（用于论文写作，2026-08-28）  
**适用范围**：LFV 第一阶段，单个或少量源演示向同类别新实例迁移连续 Contact Field；  
**与实现的关系**：本文档描述论文中希望实现和验证的完整算子，不把当前代码中尚未充分验证的工程细节写成已经完成的实验结论。

本文档不是论文 Method 章节的成稿，而是一份可直接支持论文写作、公式检查、实验设计和代码实现的技术材料。它回答四个问题：

1. 源任务知识是什么，以及它如何表示为可传输的场；
2. 语义对应和部件内部结构对应分别由什么计算完成；
3. 源场如何输运为目标实例上的连续热力；
4. 每一步借鉴了哪篇工作，哪些部分是 LFV 的组合或新增设计。

---

## 1. 方法定位与核心叙事

Stage 1 不直接预测夹爪位姿，也不学习机器人轨迹。它首先把人类演示中已经出现的
任务接触知识迁移到一个新物体实例上，输出目标实例表面的连续 Contact Field。随后，
完整点云上的抓取生成器可以把这个场用于抓取候选筛选；抓取执行和轨迹生成属于后续阶段。

整个算子可以概括为：

```text
源演示 RGB-D + 源功能部件 mask + 连续 Contact Field
                         │
              冻结 DINOv2 稠密 patch 特征
                         │
          Soft Heatmap AffCorrs：语义部件定位
                         │
             源/目标完整可见部件点云
                         │
          kNN 几何结构 + FGW 软结构对应
                         │
          输运源 Contact 概率场（不重新分割）
                         │
        目标实例连续 Contact Field + 置信度
                         │
       下游：完整点云抓取实例化 / 机器人执行
```

这里有两个互补层次：

- **语义定位（semantic localization）**：目标图像中的哪一个区域是源演示中的同一
  功能部件，例如“杯把手”“抽屉黑色把手”。该层由 AffCorrs 风格的 DINO prototype
  匹配和正、反向循环验证完成。
- **部件内结构对应（intra-part structural correspondence）**：当目标功能部件已被
  找到以后，源 Contact Field 中的高峰、低谷、两端和中心在目标部件内部应该落到
  哪里。该层由 Fused Gromov-Wasserstein（FGW）完成。

因此，最终算子不是“把源热力图复制到目标 mask”，而是

\[
\text{source task field}
\xrightarrow{\text{semantic localization}}
\text{target functional part}
\xrightarrow{\text{structural transport}}
\text{target task field}.
\]

跨实例的含义是：源、目标可以是同类别的不同物体实例，像素坐标、相机位置、局部
尺度和外观可以变化，但目标中仍存在与任务相关的功能部件。该方法不假设源目标
点云在同一相机坐标系，也不要求人为定义“左/右/中心”轴。

---

## 2. 符号、输入和输出

### 2.1 源输入

源输入来自一个清晰的人类演示帧或源实例 RGB-D 帧：

\[
I_s\in\mathbb R^{H_s\times W_s\times 3},\quad
M_s\in\{0,1\}^{H_s\times W_s},\quad
A_s\in[0,1]^{H_s\times W_s}.
\]

- `I_s`：源 RGB 图像；
- `M_s`：源功能部件的**完整可见区域**，例如整个把手，而不是仅取 Contact 峰值；
- `A_s`：源连续 Contact Field。它可以来自手—物接触证据在点云/图像上的投影，
  也可以来自此前已经生成的连续热力图。

当需要进行 FGW 时，还需要源深度 `D_s` 和相机内参 `K_s`，用来把整个可见功能
部件提升到三维。FGW 的输入不能只包含 `A_s` 高于阈值的点，否则会失去“高热区在
完整部件中的结构位置”这一信息。

### 2.2 目标输入

\[
I_t\in\mathbb R^{H_t\times W_t\times 3},\quad
M_t\in\{0,1\}^{H_t\times W_t},\quad
D_t\in\mathbb R^{H_t\times W_t},\quad K_t.
\]

`M_t` 是目标功能部件 mask。对于抽屉任务，提示词和 mask 应指向黑色把手；抽屉
箱体是参考物体，不应把整个抽屉箱体当作功能部件 mask。对于倒水任务，mask 应指向
与任务对应的杯口/把手等部件，具体取决于定义的 Contact 语义。

### 2.3 输出

主要输出为：

\[
\widehat A_t(u,v)\in[0,1],
\]

即目标相机像素坐标系中的连续热力图。为了供后续抓取模块使用，同时输出有效深度
像素的三维点和对应分数：

\[
P_t=\{p_j^t\in\mathbb R^3\}_{j=1}^{N_t},\qquad
h_t=\{h_j^t\in[0,1]\}_{j=1}^{N_t}.
\]

建议保存以下中间结果，便于消融、调试和论文可视化：

```text
source_patch_features, target_patch_features
source_heat_patch, source_prototypes, target_prototypes
forward_vote V, backward_return Q, AffCorrs score H_aff
source_points, target_points
source_geodesic D_s, target_geodesic D_t
semantic_cost M, FGW transport T
target_heat_aff, target_heat_fgw, target_heat_final
confidence components and global confidence
```

---

## 3. 参考工作和借鉴边界

### 3.1 DINO / DINOv2：无监督稠密视觉描述符

LFV 使用冻结的 DINOv2 patch token 作为跨图像语义描述符。DINOv2 的核心价值不是
提供一个任务专用分类器，而是通过自监督训练得到可迁移的视觉表征；其 patch-level
特征可用于稠密下游任务和图像中相似部件的描述。论文与官方资料：

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023：
  [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)；
- 官方项目页：[DINOv2](https://dinov2.metademolab.com/)。

LFV 不重新训练 DINOv2，也不把 DINO 特征直接当作最终 Contact 预测。DINOv2 只负责
提供跨实例的语义相似性，Contact 场仍由源演示监督并通过后续对应算子传输。

### 3.2 AffCorrs：语义区域对应与循环验证

LFV 的语义定位骨架来自 *One-Shot Transfer of Affordance Regions? AffCorrs!*：

- [论文 arXiv:2209.07147](https://arxiv.org/abs/2209.07147)；
- [OpenReview PDF](https://openreview.net/pdf?id=GeM6VUwYinO)；
- [官方代码仓库](https://github.com/RPL-CS-UCL/UCL-AffCorrs)。

AffCorrs 的关键思想是：一个源查询区域通过 DINO 描述符匹配到目标候选区域后，
还要把目标区域反向匹配回源图像，检查它是否真正落回源查询区域。正向相似度和
反向落回概率相乘，得到循环一致性分数。这个机制比单向最近邻更能抑制“目标中
外观相似但功能错误”的区域。

### 3.3 FGW：融合特征和内部结构的软传输

LFV 的结构对应阶段使用 Fused Gromov-Wasserstein。其理论基础是 Vayer et al.,
*Fused Gromov-Wasserstein Distance for Structured Objects: Theoretical Foundations
and Mathematical Properties*：

- [arXiv:1811.02834](https://arxiv.org/abs/1811.02834)；
- 图结构版本：[PMLR paper](https://proceedings.mlr.press/v97/titouan19a/titouan19a.pdf)。

FGW 同时利用跨对象节点特征代价和对象内部结构代价。特征项要求 DINO 相似的点
对应，GW 项要求“源中点对之间的距离关系”和“目标中对应点对之间的距离关系”
保持一致。输出是满足边缘约束的软传输矩阵，而不是硬的一对一匹配。

### 3.4 RGB-D 到三维 affordance：二维语义到可执行点云

把目标二维场通过深度反投影为三维点，是 2D interaction/affordance grounding
工作的共同路线。例如：

- Yang et al., *Grounding 3D Object Affordance from 2D Interactions in Images*,
  ICCV 2023：[论文 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Grounding_3D_Object_Affordance_from_2D_Interactions_in_Images_ICCV2023_paper.pdf)；
- O³Afford 项目页：[O³Afford: One-Shot 3D Object Affordance Grounding](https://o3afford.github.io/)。

LFV 沿用“视觉语义 + 深度提升到 3D”的接口，但不声称复制这些工作中的网络或
训练流程。我们的重点是：用源 Contact Field 作为可输运的连续任务场，再用 FGW
恢复功能部件内部的位置关系。

### 3.5 TAX-Pose 的关系

TAX-Pose（*Task-Specific Cross-Pose Estimation for Robot Manipulation*，
[arXiv:2211.09325](https://arxiv.org/abs/2211.09325)）说明任务相关的跨物体对应可以
支持未见物体上的关系姿态估计。它是 LFV 叙事中“功能区域应携带任务关系”的重要
动机，但 Stage 1 的迁移算子并不是 TAX-Pose 的 pose 网络，也不直接使用 TAX-Pose
的损失。

---

## 4. 统一预处理和坐标映射

### 4.1 mask 包围框、留边和 letterbox

源图和目标图必须使用相同的预处理约定。对于每一张图：

1. 从功能部件 mask 求最小包围框；
2. 按较长边扩展固定比例的 margin；
3. 等比例缩放到统一输入尺寸（例如 `518×518`）；
4. 使用 letterbox 填充到方形；
5. 保存原图像素、裁剪图像素、缩放后坐标和 DINO patch 网格之间的映射。

RGB 使用双线性插值，二值 mask 使用最近邻或面积占比阈值，连续 `A_s` 使用双线性
插值。所有 padding patch 必须剔除，避免黑边产生伪语义原型。

### 4.2 patch 网格和掩码筛选

对预处理后的图像运行冻结 DINOv2。以 ViT-S/14、`518×518` 为例，得到：

\[
F_s^{grid},F_t^{grid}\in\mathbb R^{37\times37\times384}.
\]

将 `M_s`、`M_t` 和 `A_s` 映射到同一 patch 网格，只保留 mask 内 patch：

\[
F_s=\{f_i^s\}_{i=1}^{N_s},\quad
F_t=\{f_j^t\}_{j=1}^{N_t},\quad
a_i^s\in[0,1].
\]

每个 patch descriptor 做逐向量 L2 归一化：

\[
f\leftarrow\frac{f}{\lVert f\rVert_2+\epsilon}.
\]

归一化后内积就是余弦相似度，便于与 AffCorrs 和 FGW 的语义代价统一。

---

## 5. Soft Heatmap AffCorrs：语义功能部件定位

这一部分保留 AffCorrs 的“prototype → target region → source image”循环结构，
但将原本的二值 query mask 改为连续热力加权。为便于论文叙述，下面的公式使用 LFV
记号；它们是对 AffCorrs 核心计算的连续化重写，而不是声称原论文已经使用连续
Contact heatmap。

### 5.1 源正热集合

源 Contact Field 不直接被压缩为一个二值区域。首先定义高热 patch 集合：

\[
\mathcal P_s=\{i\mid a_i^s>\tau_{pos}\}.
\]

其中 `tau_pos` 是正热阈值，例如 0.20。记录保留下来的热力质量：

\[
r_{keep}=\frac{\sum_{i\in\mathcal P_s}a_i^s}
{\sum_i a_i^s+\epsilon}.
\]

如果 `r_keep` 太低，说明源热力过于分散或 mask/坐标映射有问题，应拒绝本次迁移
或降低阈值，而不是强行聚类。

### 5.2 热力加权源 prototypes

对正热 patch 进行加权 K-Means，得到 `K_s` 个源 Contact prototypes：

\[
\min_{\{z_k^s\},\{c_i\}}
\sum_{i\in\mathcal P_s}a_i^s
\left\lVert f_i^s-z_{c_i}^s\right\rVert_2^2,
\qquad c_i\in\{1,\ldots,K_s\}.
\tag{1}
\]

每次更新中心后重新做 L2 归一化。第 `k` 个 prototype 的热力质量及其归一化权重为：

\[
m_k=\sum_{i\in\mathcal P_s,c_i=k}a_i^s,
\qquad
\omega_k=\frac{m_k}{\sum_r m_r+\epsilon}.
\tag{2}
\]

`K_s=6` 是一个小的源查询原型数。它不是目标热力图的分辨率，而是源 Contact
区域的语义摘要数量。

**来源边界**：式 (1) 对应 AffCorrs 的源 query-region prototype clustering；
式 (2) 的热力质量权重是 LFV 的连续热力扩展。

### 5.3 目标功能区域过分割

对目标 mask 内的全部 patch 做较密集的 K-Means，形成 `K_t` 个目标区域 prototype：

\[
\min_{\{z_j^t\},\{d_l\}}
\sum_{l}\left\lVert f_l^t-z_{d_l}^t\right\rVert_2^2,
\qquad d_l\in\{1,\ldots,K_t\}.
\tag{3}
\]

通常使用 `K_t=64`。过分割的目的不是直接得到 Contact mask，而是保留功能部件内
足够多的候选语义区域，以便正、反向匹配能表达部件内部差异。

### 5.4 正向区域投票

源 Contact prototype 与目标区域 prototype 的余弦相似度为：

\[
S_{kj}=\left\langle z_k^s,z_j^t\right\rangle.
\tag{4}
\]

沿目标区域做温度 Softmax：

\[
P_{kj}^{f}=
\frac{\exp(S_{kj}/\tau_f)}
{\sum_{r=1}^{K_t}\exp(S_{kr}/\tau_f)}.
\tag{5}
\]

源 Contact prototype 按热力质量投票：

\[
V_j=\sum_{k=1}^{K_s}\omega_kP_{kj}^{f}.
\tag{6}
\]

`V_j` 表示目标区域 `j` 从源 Contact 语义出发的正向支持度。低温度使匹配更集中，
但温度过低会造成单个 prototype 的偶然相似度主导结果。

### 5.5 反向落回源 Contact Field

对每个目标区域 prototype，与源 mask 内**全部源前景 patch**计算相似度：

\[
B_{ji}=\left\langle z_j^t,f_i^s\right\rangle.
\tag{7}
\]

沿源位置做温度 Softmax：

\[
P_{ji}^{b}=\frac{\exp(B_{ji}/\tau_b)}
{\sum_{r=1}^{N_s}\exp(B_{jr}/\tau_b)}.
\tag{8}
\]

将源连续 Contact 归一化成概率分布：

\[
\bar A_s(i)=\frac{a_i^s}{\sum_r a_r^s+\epsilon}.
\tag{9}
\]

目标区域向源 Contact 峰值的反向落回概率为：

\[
Q_j=\sum_{i=1}^{N_s}P_{ji}^{b}\bar A_s(i).
\tag{10}
\]

这一步继承 AffCorrs 的循环验证思想，但不再使用“是否落入二值 query mask”这一
判断，而是让落回源高热位置的概率获得更大权重。

### 5.6 语义区域分数和像素热力

最终 Soft Heatmap AffCorrs 区域分数为：

\[
H_j^{aff}=V_jQ_j.
\tag{11}
\]

将 `H_j^{aff}` 分配给属于目标 cluster `j` 的全部 patch，通过保存的逆映射插值
回目标原图，乘以 `M_t`，并做 min-max 归一化：

\[
A_{aff}(u,v)=
M_t(u,v)\cdot
\operatorname{Norm}\left[\operatorname{Interp}^{-1}(H_{d(u,v)}^{aff})\right].
\tag{12}
\]

LFV 明确不使用 AffCorrs 原流程中的 CRF 后处理，也不将结果二值化。连续输出保留
后续 FGW 输运所需要的强弱关系，例如把手中央高、两端低。

**重要理解**：式 (11) 主要解决“哪个功能部件”，不能保证部件内部的热力峰位置
在不同实例上保持几何对应。因此 `A_aff` 是语义定位结果，不是最终 Contact Field。

---

## 6. RGB-D 提升和功能部件内部图结构

FGW 的输入必须是源、目标功能部件的完整可见点集，而不是只取高热点。

### 6.1 深度反投影

对 mask 内有效深度像素 `u,v`，设深度值为相机坐标系的 `z`，内参为：

\[
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
\]

反投影公式为：

\[
p(u,v)=D(u,v)K^{-1}[u,v,1]^\top
=\left(
\frac{(u-c_x)z}{f_x},
\frac{(v-c_y)z}{f_y},z
\right)^\top.
\tag{13}
\]

源和目标分别得到：

\[
P_s=\{p_i^s\}_{i=1}^{N_s},\qquad P_t=\{p_j^t\}_{j=1}^{N_t}.
\]

源二维热力通过同一像素/patch 映射赋给三维点，得到 `h_i^s`。目标 DINO patch
特征通过像素所属 patch 或双线性插值得到每个目标三维点的 `f_j^t`。

### 6.2 下采样和索引一致性

完整可见部件点数可能远大于 FGW 的计算预算。先用 voxel 或 FPS 下采样到约
256–512 个节点，记下采样索引和原始点索引映射。这里的 FGW 节点数与 AffCorrs
的 `K_t=64` 不同：前者是三维结构图节点数，后者是二维语义区域 cluster 数。

如果后续需要恢复到原始目标点云，必须保存每个原始点对应的最近下采样节点或 3NN
邻域，不能重新随机采样，否则热力和几何点会失去对应关系。

### 6.3 kNN 图和 geodesic 距离

在源、目标下采样点云上分别建立 kNN 图（建议 `k=8–12`）。边权为欧氏边长：

\[
w_{ik}^s=\lVert p_i^s-p_k^s\rVert_2,
\qquad
w_{jl}^t=\lVert p_j^t-p_l^t\rVert_2.
\tag{14}
\]

对图运行全源点对最短路，得到近似部件表面 geodesic 距离矩阵：

\[
D_s(i,k)=\operatorname{ShortestPath}_{G_s}(i,k),
\quad
D_t(j,l)=\operatorname{ShortestPath}_{G_t}(j,l).
\tag{15}
\]

为了去除相机尺度和不同实例大小的影响，用非零距离的中位数、分位数或直径归一化：

\[
\widetilde D_s=\operatorname{clip}\left(\frac{D_s}{q_s+\epsilon},0,d_{max}\right),
\qquad
\widetilde D_t=\operatorname{clip}\left(\frac{D_t}{q_t+\epsilon},0,d_{max}\right).
\tag{16}
\]

其中 `q_s,q_t` 可取非零距离的 median 或 q95。归一化使结构项关注相对部件几何，
而不是绝对相机坐标或人工左右轴。

---

## 7. Fused Gromov-Wasserstein 结构传输

### 7.1 语义跨对象代价

对源节点 `i` 和目标节点 `j`，由 DINO 特征构造语义代价：

\[
C_{ij}^{sem}=1-\left\langle f_i^s,f_j^t\right\rangle.
\tag{17}
\]

如果需要使用 AffCorrs 的循环结果作为软置信度，可以定义源/目标点对应的循环
一致性代价 `C_ij^{cycle}`，并加入：

\[
C_{ij}^{sem}\leftarrow C_{ij}^{sem}
+\lambda_c(1-C_{ij}^{cycle}).
\tag{18}
\]

式 (18) 是可选项。第一版可以先只使用 DINO 语义代价，避免把二维 cluster 分数和
三维点级代价重复计算；加入循环代价应作为消融实验，而不是默认的隐式必要步骤。

### 7.2 FGW 传输矩阵和边缘约束

令源、目标节点质量均匀：

\[
a_i=\frac1{N_s},\qquad b_j=\frac1{N_t}.
\tag{19}
\]

可行传输集合为：

\[
\Pi(a,b)=\left\{T\ge0\mid
T\mathbf 1=a,\;T^\top\mathbf1=b\right\},
\tag{20}
\]

其中 `T[i,j]` 表示源节点 `i` 向目标节点 `j` 的软传输质量。

标准 FGW 的离散目标可以写成：

\[
\min_{T\in\Pi(a,b)}
(1-\alpha)\sum_{i,j}C_{ij}^{sem}T_{ij}
 +\alpha\sum_{i,k,j,l}
\left(\widetilde D_s(i,k)-\widetilde D_t(j,l)\right)^2
T_{ij}T_{kl}.
\tag{21}
\]

也可以将结构项写成四阶张量 `L[i,k,j,l]` 与 `T⊗T` 的内积。`alpha` 控制语义和
结构的平衡：

- `alpha=0`：退化为仅使用语义特征的 Wasserstein/OT 匹配；
- `alpha=1`：接近仅使用内部结构的 GW 匹配；
- 第一版建议从 `alpha=0.5` 开始，并用 `0.2/0.5/0.8` 做消融。

使用 POT 等库的 entropic FGW/Sinkhorn 求解器可以得到软矩阵 `T`。熵正则应记录在
配置和 checkpoint 中，因为正则过大时会过度平滑 Contact 峰，过小时则容易陷入
局部硬匹配。

**来源边界**：式 (17)–(21) 的语义—结构融合和边缘约束来自 FGW/GW 文献；“把源
Contact 概率直接作为待输运信号”是 LFV 在 affordance 场景中的任务化使用，不是
AffCorrs 原论文的输出。

### 7.3 直接输运 Contact Field

FGW 求出 `T` 后，不再由 `T` 重新做目标部件分割，也不把输运矩阵的最大列索引
当作硬匹配。对每个目标下采样点直接计算源 Contact 概率的条件平均：

\[
h_j^{fgw}=
\frac{\sum_{i=1}^{N_s}T_{ij}h_i^s}
{\sum_{i=1}^{N_s}T_{ij}+\epsilon}.
\tag{22}
\]

由于 `T` 具有目标边缘约束，分母在理想数值情况下接近 `b_j`；显式保留分母可以
提高不同求解器、非均匀质量或截断矩阵下的稳定性。

将下采样点热力恢复到目标原始功能部件点云：

\[
h_q=\frac{\sum_{r\in\mathcal N_3(q)}
\frac{h_r^{fgw}}{\lVert p_q-p_r\rVert_2+\epsilon}}
{\sum_{r\in\mathcal N_3(q)}
\frac1{\lVert p_q-p_r\rVert_2+\epsilon}}.
\tag{23}
\]

必要时只做轻量 kNN 平滑；平滑半径和权重必须记录，避免把结构传输结果人为扩散
到整个部件。最后按照目标点到像素的索引把 `h_q` 恢复为二维图 `A_fgw`。

---

## 8. 语义门控和最终目标场

FGW 依赖结构和特征，理论上可能在语义部件之外产生少量错误对应；Soft Heatmap
AffCorrs 则提供一个“目标确实属于源功能部件”的语义门。将 AffCorrs 图归一化为
`G_t`，构造门控：

\[
G_t(u,v)=\left(A_{aff}(u,v)\right)^\gamma.
\tag{24}
\]

一种稳健的最终融合为：

\[
A_{final}(u,v)=
M_t(u,v)\,A_{fgw}(u,v)
\left[\lambda+(1-\lambda)G_t(u,v)\right].
\tag{25}
\]

最后在 `M_t` 内归一化到 `[0,1]`。其中：

- `lambda=0`：严格用 AffCorrs 作为语义门；
- `lambda>0`：保留 FGW 的部件内部结构峰，降低 AffCorrs 误差对最终场的破坏；
- `gamma` 控制门控的尖锐程度。

这不是“二次分割”，而是把语义定位作为 FGW 场的软置信度。论文中应明确：
AffCorrs 决定**在哪里传输**，FGW 决定**部件内部如何传输**。

---

## 9. 迁移置信度、拒绝机制和多源扩展

### 9.1 循环一致性置信度

源热力归一化为 `\bar A_s` 后，均匀反向基线和理论上限为：

\[
q_0=1/N_s,\qquad q_{max}=\max_i\bar A_s(i).
\]

可将 `Q_j` 线性校准到 `[0,1]`：

\[
\widetilde Q_j=\operatorname{clip}
\left(\frac{Q_j-q_0}{q_{max}-q_0+\epsilon},0,1\right).
\tag{26}
\]

区域循环置信度可按 `V_j\widetilde Q_j` 的质量加权平均得到：

\[
C_{cycle}=\sum_j\pi_j\widetilde Q_j,
\qquad
\pi_j=\frac{V_j}{\sum_rV_r+\epsilon}.
\tag{27}
\]

### 9.2 峰值显著度和热力熵

定义目标场在部件内的 q50、q95，峰值显著度例如为：

\[
C_{peak}=\operatorname{clip}
\left(\frac{q_{95}-q_{50}}{q_{95}+\epsilon},0,1\right).
\tag{28}
\]

把离散目标热力归一化为概率 `p_j`，计算归一化熵：

\[
H(p)=-\frac{1}{\log N_t}\sum_jp_j\log(p_j+\epsilon),
\qquad
C_{entropy}=1-H(p).
\tag{29}
\]

熵过高表示热力近似铺满整个部件，熵过低则可能是单个异常点。可结合峰值显著度、
有效点比例和 FGW 求解状态构造全局置信度，例如几何平均：

\[
C_{global}=
\left(C_{cycle}C_{peak}C_{entropy}C_{valid}\right)^{1/4}.
\tag{30}
\]

当 `C_global < tau_accept` 时返回“拒绝迁移”状态，并保留中间结果用于调试；不要
将低置信度结果静默地送入抓取模块。

### 9.3 多源演示扩展

第一版使用一个源清晰帧。稳定后，同一演示的多个源帧可以分别运行完整算子，得到
`A_final^(m)` 和 `C_global^(m)`，再按置信度融合：

\[
A_{multi}(u,v)=
\frac{\sum_m C_m A_{final}^{(m)}(u,v)}
{\sum_m C_m+\epsilon}.
\tag{31}
\]

多帧扩展可以覆盖不同可见角度，但应在论文中说明：它是源任务知识的多证据融合，
不是把目标单视角观测伪装成完整点云。目标端仍只使用当次 RGB-D 可见部件；不可见
表面补全属于执行侧独立模块。

---

## 10. 为什么必须先 AffCorrs 再 FGW

直接对完整物体点云做 FGW 会让几何相似区域互相竞争，尤其是多个圆柱、平面或重复
把手结构。先用 AffCorrs 做功能部件定位有三个作用：

1. 将 OT 的搜索域限制在“同一功能语义部件”；
2. 用反向循环一致性排除语义相似但任务无关的区域；
3. 让 FGW 只负责部件内部的相对结构，而不是同时承担“找部件”和“找 Contact 峰”。

反过来，只用 AffCorrs 也不够：它的区域 prototype 主要回答“目标有哪一个对应
部件”，不显式最小化源、目标内部点对距离关系，因而容易把源把手中央的高热扩散到
整个目标把手。两者组合形成职责分离：

```text
AffCorrs: semantic part correspondence
FGW:     intra-part structural correspondence
Transport: source continuous Contact field → target continuous Contact field
```

---

## 11. 与原论文相比，哪些是继承、组合和新增

| 组件 | 参考来源 | LFV 中的具体用法 | 论文表述边界 |
|---|---|---|---|
| 冻结稠密 patch 特征 | DINO / DINOv2 | 源、目标 patch descriptor，逐向量 L2 归一化 | DINOv2 是通用视觉表征，不是 Contact 监督 |
| 源 query prototype | AffCorrs | 对源高热 patch 做 prototype clustering | 原论文是二值 query；热力加权是 LFV 扩展 |
| 目标区域过分割 | AffCorrs | 目标功能 mask 内 `K_t=64` 个区域 prototype | 用于候选区域表达，不等于最终 Contact mask |
| 正向 prototype→region | AffCorrs | 余弦相似度 + 温度 Softmax + 热力质量投票 | Soft vote 是连续化改造 |
| 反向 region→source | AffCorrs | 对源全部前景做 Softmax，按源连续热力落回 | 将二值 query membership 改为连续概率 |
| 循环分数 | AffCorrs | `H_aff=V·Q`，不做 CRF、不二值化 | 语义定位分数，不是最终结构场 |
| RGB-D 反投影 | 2D→3D affordance 工作 | 将目标二维场和 DINO 特征对齐到点云 | 工程接口，不能归因于单一网络论文 |
| 内部 kNN/geodesic | GW/FGW 结构建模 | 部件内部相对距离矩阵 | 不使用 PCA、OBB 或人为左右轴 |
| FGW 软传输 | Vayer et al. | 融合 DINO 语义代价和 geodesic 结构代价 | FGW 是优化器；Contact 场输运是 LFV 任务化应用 |
| 直接场输运 | LFV 新增 | `h_t=T^T h_s / T^T1` | 不从 `T` 重新分割目标部件 |
| 语义门控 | LFV 工程设计 | AffCorrs 作为 FGW 场的软 gate | 应通过消融证明是否必要 |
| 多源置信度融合 | 受多源对应思想启发 | 按迁移置信度融合不同源帧 | 第一版可不实现 |

这张表应作为论文写作时的“贡献边界检查表”，避免把外部模块简单堆叠后声称为同一
篇论文中的原始设计，也避免把 LFV 的连续热力扩展错误归因给 AffCorrs。

---

## 12. 一个完整样例的张量/计算流程

下面给出从输入到输出的一次迁移，便于实现和画方法图：

```text
1. 读取 source_rgb, source_mask, source_heatmap
2. 读取 target_rgb, target_mask, target_depth, K_target
3. 源/目标 bbox + margin + resize + letterbox
4. 冻结 DINOv2:
      Fs_grid, Ft_grid: [37, 37, D]
5. mask/heatmap 映射到 patch 网格:
      Fs: [Ns, D], Ft: [Nt_patch, D], a_s: [Ns]
6. 源高热 patch weighted K-Means:
      source_proto: [Ks, D], omega: [Ks]
7. 目标 patch K-Means:
      target_proto: [Kt, D], target_cluster_id: [Nt_patch]
8. AffCorrs:
      S [Ks, Kt] → Pf [Ks, Kt] → V [Kt]
      B [Kt, Ns] → Pb [Kt, Ns] → Q [Kt]
      H_aff [Kt] = V * Q
      A_aff: [Ht, Wt]
9. 源/目标功能部件 RGB-D 反投影:
      Ps [Ns3, 3], Pt [Nt3, 3]
      hs [Ns3], fs [Ns3, D], ft [Nt3, D]
10. FPS/voxel:
      Ps_ds [Ns', 3], Pt_ds [Nt', 3], Ns',Nt'≈256–512
11. kNN + shortest path:
      Ds [Ns', Ns'], Dt [Nt', Nt']
12. FGW:
      C_sem [Ns', Nt'], T [Ns', Nt']
13. 直接输运:
      h_fgw [Nt'] = T^T hs / (T^T 1)
14. 3NN 恢复原始目标部件点云和二维像素:
      A_fgw [Ht, Wt]
15. 语义门控和归一化:
      A_final = mask * A_fgw * (lambda + (1-lambda) * A_aff^gamma)
16. 置信度:
      C_cycle, C_peak, C_entropy, C_valid → C_global
17. 保存结果和可视化:
      source heat, AffCorrs heat, FGW heat, final heat, point cloud heat
```

---

## 13. 推荐配置和消融变量

建议把下面参数写入 YAML，而不是散落在代码中：

```yaml
feature:
  backbone: dinov2_vits14
  input_size: 518
  l2_normalize: true
  freeze: true

affcorrs:
  tau_pos: 0.20
  source_clusters: 6
  target_clusters: 64
  tau_forward: 0.10
  tau_backward: 0.05

fgw:
  enabled: true
  nodes: 384
  knn_k: 10
  alpha: 0.50
  semantic_cost: cosine
  geodesic_normalization: q95
  entropy_regularization: 0.01
  interpolation_k: 3

fusion:
  semantic_gate: true
  lambda_floor: 0.05
  gamma: 0.5

confidence:
  accept_threshold: 0.35
```

最小消融集合：

1. AffCorrs only vs. AffCorrs + direct nearest-neighbor heat transfer；
2. AffCorrs + FGW，`alpha∈{0.2,0.5,0.8}`；
3. 原始二值/未加权 source prototype vs. 连续热力加权 prototype；
4. 不使用语义 gate vs. 使用式 (25)；
5. 全部功能部件点云 vs. 仅高热点输入 FGW（后者应作为失败对照）；
6. `K_t∈{32,64,96}`，验证目标过分割分辨率对峰值保持的影响。

评价指标至少包括：

- Contact MSE、Soft-IoU 或 AUPRC；
- 预测 Contact 峰值到 GT 区域的三维/二维距离；
- 目标部件 mask 内外能量比；
- 热力熵与峰值显著度；
- 迁移拒绝率和低置信度误迁移率；
- 在完整点云上由热力引导的抓取成功率（作为下游验证，不作为 Stage 1 唯一指标）。

---

## 14. 论文中应明确的限制和避免的表述

### 14.1 观测范围

Stage 1 的目标热力首先是**目标当前 RGB-D 可见功能部件上的 Contact Field**。它不
自动恢复单视角看不到的背面，也不等价于完整物体的全表面 affordance。若执行侧使用
完整点云或 SAM3D/多视角重建，应明确这是后端几何补全和抓取生成，不是本迁移算子的
语义对应输入。

### 14.2 源 mask 的粒度

源 mask 必须覆盖整个功能部件，源 Contact Field 才能在部件内部表达相对位置。只把
高热 Contact 峰作为 FGW 的点云输入会让结构匹配失去参照系；只把整个物体作为 mask
则会稀释功能语义并增加错误对应。

### 14.3 连续场和二值区域的区别

AffCorrs 输出的 `A_aff` 只表示语义对应区域；`A_final` 才是用于抓取候选选择的
连续任务 Contact Field。论文中应避免将“AffCorrs 区域分数”“FGW 输运热力”和
“机器人最终抓取概率”混称为同一个量。

### 14.4 不应声称的内容

- 不应说 DINOv2 本身学习了 Contact；
- 不应说 AffCorrs 原论文使用了连续热力加权和 FGW；
- 不应说 FGW 自动补全了目标不可见表面；
- 不应说只要语义相似就一定得到正确抓取，必须报告置信度和失败样例；
- 不应把 TAX-Pose 的跨物体 pose 估计公式当作本算子的训练目标。

---

## 15. 可直接用于论文的贡献表述素材

下面不是 Method 成稿，而是可改写成摘要、引言或贡献点的技术事实：

1. 我们把演示中的接触信息表示为连续、可输运的 Contact Field，而不是只保留一个
   二值功能区域；
2. 我们将语义区域定位和功能部件内部结构对应解耦：前者采用 DINOv2 特征与
   AffCorrs 风格循环匹配，后者采用融合语义与 geodesic 结构的 FGW；
3. 我们不从 FGW 传输矩阵重新进行目标分割，而是直接输运源 Contact 概率场，因此
   源把手中央的高热/两端低热等相对分布能够在目标同类部件内部保留；
4. 我们把二维语义—接触场通过 RGB-D 提升为目标部件点云场，为后续完整点云抓取
   生成提供统一接口；
5. 我们输出循环一致性、峰值显著度和热力熵组成的迁移置信度，在语义或结构对应
   不可靠时可以拒绝下游执行，而不是强制生成抓取。

上述表述的创新性应建立在对照实验上：至少与 AffCorrs-only、最近邻热力复制、
无结构项 OT 和仅高热点 FGW 对比，并在不同实例、不同视角和不同部件类型上报告结果。

---

## 16. 参考文献清单

1. Maxime Oquab et al. **DINOv2: Learning Robust Visual Features without Supervision**.
   arXiv:2304.07193, 2023. [论文](https://arxiv.org/abs/2304.07193)
2. K. Sharma et al. **One-Shot Transfer of Affordance Regions? AffCorrs!**
   arXiv:2209.07147. [论文](https://arxiv.org/abs/2209.07147) ·
   [代码](https://github.com/RPL-CS-UCL/UCL-AffCorrs)
3. Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, Nicolas Courty.
   **Fused Gromov-Wasserstein Distance for Structured Objects: Theoretical Foundations
   and Mathematical Properties**. arXiv:1811.02834, 2018/2019.
   [论文](https://arxiv.org/abs/1811.02834)
4. Titouan Vayer et al. **Optimal Transport for Structured Data with Application on
   Graphs**. PMLR 97, 2019. [论文](https://proceedings.mlr.press/v97/titouan19a/titouan19a.pdf)
5. Y. Yang et al. **Grounding 3D Object Affordance from 2D Interactions in Images**.
   ICCV 2023. [论文 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Grounding_3D_Object_Affordance_from_2D_Interactions_in_Images_ICCV2023_paper.pdf)
6. **O³Afford: One-Shot 3D Object Affordance Grounding**. CoRL 2025.
   [项目页](https://o3afford.github.io/)
7. R. M. W. et al. **TAX-Pose: Task-Specific Cross-Pose Estimation for Robot
   Manipulation**. arXiv:2211.09325. [论文](https://arxiv.org/abs/2211.09325)

---

## 17. 一句话总结

LFV 的跨实例迁移算子先用 DINOv2 + AffCorrs 风格的正、反向循环匹配确定“目标中哪
个功能部件与源任务对应”，再把源、目标整个可见功能部件提升为带 DINO 特征的三维
图，用 FGW 保持部件内部的相对结构，最后直接将源连续 Contact Field 通过软传输
矩阵输运到目标点云和图像；因此语义定位、结构定位和任务场输出各自职责清晰，既
保留了参考工作的核心公式，又明确区分了 LFV 为连续热力和结构传输增加的部分。
