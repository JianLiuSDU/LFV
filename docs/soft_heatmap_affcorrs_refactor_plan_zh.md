# LFV Soft Heatmap AffCorrs 重构、计算流程与实施记录

状态：第一版已实现并通过真实 `episode_0 → ManiSkill` GPU 验证。

日期：2026-07-31

> 边界说明：本文只定义并验收纯二维 Soft Heatmap AffCorrs 基线，因此下文所有
> “不读取 depth、不调用 GraspNet”的描述仍然成立。其输出现在已有一个严格解耦
> 的可选下游阶段，用 ManiSkill 深度与完整 mesh 做三维提升、反平行表面传播和
> top-down GraspNet 抓取；实现与结果见
> `docs/transferred_heat_topdown_grasp_zh.md`。下游不会反向修改二维迁移分数。
> 2026-08-03 又增加了显式配置启用的 RGB-D AffCorrs+FGW 结构迁移变体；它不会
> 改写本基线，实现、数据契约、A/B 结果与限制见
> `docs/affcorrs_fgw_contact_transfer_zh.md`。

实际结果：

```text
accepted = true
global confidence = 0.336578
cycle consistency = 0.098644
peak score = 0.999895
entropy = 0.613426
retained source heat mass = 0.897586
```

固定可视化中，目标连续热力正确落在仿真杯子左侧把手区域，未落到杯身或旁边
的碗。实现代码、命令、输出与删除记录见第 10 至 14 节；当前仓库总览另见
`docs/project_architecture_and_development_guide_zh.md`。

## 1. 本轮重构的目标

LFV 原 Stage-1 Joint Contact–Grasp Diffusion 依赖大量同分布训练数据，但当前
hand-pouring 数据在物体实例、外观、视角和接触模式上都过于单一。继续训练一个
逐点 Contact/Grasp 生成模型，容易记忆杯子实例和相机布局，不能可靠验证同类别
新实例上的接触位置泛化。

本轮将 Stage 1 改为训练自由的 one-shot 接触区域迁移：

```text
人类演示中的清晰源帧
    source_rgb + source_mask + source_heatmap
                         |
                         v
冻结的稠密语义特征 + Soft Heatmap AffCorrs
                         |
                         v
仿真新实例目标 RGB 视角上的连续 affordance 热力
```

第一版只实现“一个源帧到一个目标图像”的确定性闭环，不训练网络，不引入 CRF，
不做 DIFT 融合，不做多帧融合。第一版不读取目标 depth 和相机内参，不输出点云，
不做完整点云双侧传播，不调用或修改 GraspNet，也不以抓取成功作为验收条件。
新方法的核心名称为 **Soft Heatmap AffCorrs**。

### 1.1 当前二维里程碑

```text
episode_0 的源 RGB/mask/连续热力
    -> Soft Heatmap AffCorrs
    -> ManiSkill 杯子截图视角上的二维连续热力
    -> transfer_result.npz + transfer_report.json + transfer_summary.png
```

当前结果只回答一个问题：

> 源杯子把手上的连续 affordance 热力，能否依靠冻结语义特征和循环一致性，
> 正确迁移到仿真新杯子实例的把手区域？

点云、不可见侧、抓取姿态和机器人执行都属于后续独立里程碑，不能混入本轮失败
分析。

## 2. 文献依据与方法边界

### 2.1 AffCorrs 提供的核心结构

[AffCorrs 论文](https://proceedings.mlr.press/v205/hadjivelichkov23a/hadjivelichkov23a.pdf)
和[官方实现](https://github.com/RPL-CS-UCL/UCL-AffCorrs)的原始流程是：

1. 使用预训练 DINO-ViT 提取源图和目标图的稠密描述符；
2. 将源图二值查询区域聚类成少量 query centroids；
3. 将目标显著前景过分割成较多 target centroids，官方实现还单独聚类背景；
4. query centroids 正向匹配 target centroids，并对 query 投票求和；
5. target centroids 反向匹配源图全部描述符；
6. 在源图二值 query mask 内累计反向概率；
7. 正向分数与反向概率相乘形成循环一致性分数；
8. 将目标 centroid 分数映射回像素，再由 CRF 输出二值区域。

官方代码的反向匹配不是只返回源查询区域，而是先对源图完整描述符做 Softmax，
再在查询 mask 内求概率质量。这正是排除“目标区域其实更像源物体其他部位”的
关键步骤。

### 2.2 LFV 的内部改动

本项目保留上述“源查询聚类—目标过分割—正向投票—反向全源验证—乘积循环
分数”的骨架，做以下明确改动：

1. 二值源查询 mask 改为连续接触热力；
2. 源查询 K-Means 改为热力加权 K-Means；
3. 源原型对目标区域的投票改为热力质量加权；
4. 反向落回源查询区域的二值求和改为连续热力加权；
5. 不使用 CRF，不输出二值 mask，始终保留连续目标热力；
6. 使用 DINOv2 替代论文中的 DINO-ViT；
7. 增加可解释的置信度和拒绝迁移机制；
8. 已有目标实例 mask 时只聚类 mask 内前景，mask 外直接置零，不再建立
   AffCorrs 原实现中的背景 clusters；
9. 源和目标都使用物体 bbox crop、等比例 resize、对称 letterbox 和显式可逆
    坐标映射。

因此，不能把连续热力公式、DINOv2 或置信度设计表述成 AffCorrs 原论文的贡献；
它们是 LFV 在 AffCorrs 骨架上的扩展。

[DINOv2](https://arxiv.org/abs/2304.07193)提供冻结的通用视觉特征，官方
[代码与权重](https://github.com/facebookresearch/dinov2)支持直接提取 patch
tokens。LFV 第一版使用仓库已有的 ViT-S/14 权重，DINOv2 是工程升级，而不是
AffCorrs 原始设计的一部分。

[DIFT](https://arxiv.org/abs/2306.03881)证明预训练扩散模型的中间特征无需
任务微调即可产生语义、几何和时间对应关系。它适合作为后续可替换特征后端或与
DINOv2 分数融合，但第一版不引入 Stable Diffusion 依赖和额外超参数。

[RAM](https://arxiv.org/abs/2407.04689)采用“检索/迁移二维 affordance，再利用
RGB-D 提升到三维执行表示”的路线。它只作为未来从二维结果走向机器人执行的参考；
RAM 的检索记忆、三维 lifting、轨迹方向和抓取模块都不在当前里程碑。

[Bi-Adapt](https://biadapt-project.github.io/)使用视觉基础模型完成跨类别
contact-point mapping，并利用支持集进行后续适配。它支持未来的多源候选与融合
方向，但不应被描述为本项目单帧循环匹配公式的直接来源。

[O³Afford](https://arxiv.org/abs/2509.06233)将 DINOv2 语义与点云局部几何结合，
说明视觉语义和三维几何互补。它只用于指导未来扩展；当前第一版完全停在二维
目标图像热力，不加入点云几何。

## 3. 已确认可直接使用的 LFV 数据

### 3.1 源样本

第一版固定源样本：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0
```

读取：

```text
source_rgb:
    rgb zarr 的 frame 39
    shape = [480, 640, 3], uint8, RGB

source_mask:
    sam_mask/affordance_mask.npy
    shape = [480, 640], bool

source_heatmap:
    contact_heatmap/contact_heatmap.npz["heatmap_2d"]
    shape = [480, 640], float32, range [0, 1]
```

该热力图的峰值位于源杯子把手上，适合作为单源 one-shot query。

已有的
`dinov2_features/anchor_dinov2_grid.npz` 是在完整 `640×480` 图像上采用右/下
补齐得到的 `35×46×384` 特征。它没有统一的物体 bbox crop、对称 letterbox 和
完整可逆映射，因此新算法不能直接把它和目标特征混用。第一版会对源图和目标图
通过同一个新预处理器重新提取特征；旧网格只作为数据完整性参考。

### 3.2 目标样本

第一版固定目标：

```text
/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/
seed_0_dataset_aligned/pouring_snapshot.npz
```

读取：

```text
target_rgb           = rgb           [480, 640, 3] uint8 RGB
target_mask          = cup_mask      [480, 640] uint8/bool
```

该快照中杯子位于图像左侧，把手完整朝向图像左方，和源 episode 的布局接近，
适合先隔离并验证语义迁移本身。

虽然快照文件还包含 `depth_m` 和 `intrinsic_cv`，当前 adapter 不读取这两个键，
避免二维迁移测试无意依赖三维信息。

## 4. 稳定数据接口

### 4.1 输入接口

```python
SourceContactExample:
    rgb: np.ndarray              # [Hs, Ws, 3], uint8, RGB
    mask: np.ndarray             # [Hs, Ws], bool
    heatmap: np.ndarray          # [Hs, Ws], float32, [0, 1]
    sample_id: str

TargetObservation:
    rgb: np.ndarray              # [Ht, Wt, 3], uint8, RGB
    mask: np.ndarray             # [Ht, Wt], bool
    sample_id: str
```

输入检查必须拒绝：

- RGB、mask、heat 空间尺寸不一致；
- mask 为空；
- heatmap 含 NaN/Inf 或超出 `[0,1]`；
- 源正热 patch 数少于请求的源聚类数。

### 4.2 输出接口

```python
TransferResult:
    target_heatmap: np.ndarray          # [Ht, Wt], float32, [0, 1]
    target_heatmap_raw: np.ndarray      # [Ht, Wt], float32
    confidence: dict[str, float]
    accepted: bool
    rejection_reasons: list[str]
    diagnostics: dict[str, Any]
```

保存格式：

```text
transfer_result.npz
    target_heatmap
    target_heatmap_raw

transfer_report.json
    输入来源、所有参数、crop 映射、聚类统计、置信度和拒绝原因

transfer_summary.png
    固定的一张对比与诊断图
```

## 5. 统一预处理和坐标映射

源图和目标图分别计算自己的物体 bbox，但必须执行完全相同的规则。

### 5.1 Object-centric crop

1. 从 mask 计算紧致 `bbox_xyxy`；
2. 按 bbox 宽高的固定比例向四周扩展，第一版建议 `margin=0.15`；
3. 扩展 bbox 截断到原图范围；
4. 保留扩展后的矩形长宽比，不做非等比例拉伸；
5. 等比例缩放到 `518×518` 画布内；
6. 四周对称 padding，padding 颜色使用相同的图像均值策略；
7. 保存从原图到网络输入的仿射映射和逆映射。

`518` 能被 ViT-S/14 的 patch size `14` 整除，对应 `37×37` 特征网格。

### 5.2 同步变换

同一个映射必须应用于：

- RGB：双线性；
- 连续 heatmap：双线性；
- mask：最近邻或面积占比后阈值化；
- content-valid mask：标记真实 crop 内容，排除 padding patch。

每个 patch 使用其中心点：

\[
x_{\mathrm{in}}=(c+0.5)p,\qquad
y_{\mathrm{in}}=(r+0.5)p
\]

通过保存的逆映射恢复到原始图像坐标。需要同时保存：

```text
original_hw
raw_bbox_xyxy
expanded_bbox_xyxy
scale
resized_hw
padding_ltrb
input_hw
patch_size
grid_hw
patch_centers_original_uv
valid_content_grid
foreground_grid
```

这样输出不依赖“猜测 DINO 网格对应哪个原图像素”，也可以对预处理做精确的
round-trip 单元测试。

## 6. DINOv2 稠密描述符

第一版配置：

```text
backend: timm
model: vit_small_patch14_dinov2
weights: third_party/dinov2_weights/dinov2_vits14_pretrain.pth
input: [1, 3, 518, 518]
output grid: [37, 37, 384]
training: disabled
```

处理要求：

1. 模型只加载一次；
2. `eval()` 和 `torch.no_grad()`；
3. 只取 patch tokens，排除 CLS/register tokens；
4. 将每个 patch descriptor 做 L2 归一化；
5. padding patch 不参与任何聚类和匹配；
6. 源和目标必须使用同一模型、同一层、同一归一化和同一输入尺寸；
7. 缓存必须包含权重路径、权重 hash、预处理配置和映射，配置不一致时禁止复用。

接口设计为特征后端协议：

```python
class DenseFeatureExtractor(Protocol):
    def extract(
        self,
        rgb_input: np.ndarray,
    ) -> np.ndarray:
        """Return L2-normalized [Gh, Gw, D] descriptors."""
```

以后加入 DIFT 时只新增实现，不修改 Soft AffCorrs 数学模块。

## 7. Soft Heatmap AffCorrs 计算流程

### 7.1 网格变量

在源前景 patch 上记：

\[
F_s=\{f_i^s\}_{i=1}^{N_s},\qquad
A_s=\{a_i^s\}_{i=1}^{N_s}
\]

其中 `f_i^s` 已 L2 归一化，`a_i^s∈[0,1]` 是变换到特征网格的连续热力。

定义正热集合：

\[
\mathcal P=\{i\mid a_i^s>\tau_{\mathrm{pos}}\}.
\]

第一版建议从 `τ_pos=0.20` 开始，并记录：

\[
r_{\mathrm{kept}}=
\frac{\sum_{i\in\mathcal P}a_i^s}
{\sum_{i=1}^{N_s}a_i^s}.
\]

如果阈值丢掉过多热力质量，迁移应拒绝或降低阈值，而不是静默继续。

### 7.2 热力加权源聚类

对 `i∈P` 执行加权 K-Means：

\[
\min_{\{z_k^s,c_i\}}
\sum_{i\in\mathcal P}
a_i^s\left\|f_i^s-z_{c_i}^s\right\|_2^2 .
\]

每次更新后对原型再次 L2 归一化。源原型权重定义为：

\[
\omega_k=
\frac{\sum_{i\in\mathcal P:c_i=k}a_i^s}
{\sum_{i\in\mathcal P}a_i^s},
\qquad
\sum_k\omega_k=1.
\]

这里把原描述中 `∑i` 的求和域明确为正热集合 `P`。如果分母使用全部前景热力，
而聚类只覆盖 `A_s>τ_pos` 的 patch，则 `∑kωk<1`，正向投票尺度会随阈值变化，
不利于样本间置信度比较。被阈值排除的热力质量由 `r_kept` 单独记录。

第一版建议：

```text
K_source = 6
n_init = 8
max_iter = 100
seed = 0
```

当正 patch 少于 6 时不自动重复点，而是显式拒绝或由配置降低 K。

### 7.3 目标前景过分割

在目标 mask 内且不属于 padding 的 patch 描述符上执行普通 K-Means：

\[
\min_{\{z_j^t,c_n^t\}}
\sum_n\left\|f_n^t-z_{c_n^t}^t\right\|_2^2.
\]

第一版建议 `K_target=64`，并保留从每个目标 foreground patch 到聚类编号的映射。
第一版不把像素坐标、法向或深度拼入聚类特征，以便单独验证语义描述符；如果目标
cluster 空间上严重碎裂，再增加“空间紧致性”消融，而不是悄悄改变基线。

为避免新增 scikit-learn 依赖，计划在 NumPy/Torch 中实现确定性的 weighted
K-Means++：

- 源初始化概率正比于 `a_i d_i²`；
- 目标初始化概率正比于 `d_i²`；
- 空簇重置为加权残差最大的点；
- 多次初始化选择加权 inertia 最低的一次。

### 7.4 正向区域投票

源接触原型和目标区域原型都已归一化，因此余弦相似度为：

\[
S_{kj}=(z_k^s)^\top z_j^t.
\]

沿目标区域做温度 Softmax：

\[
P^f_{kj}=
\frac{\exp(S_{kj}/T_f)}
{\sum_{\ell=1}^{K_t}\exp(S_{k\ell}/T_f)}.
\]

热力加权投票：

\[
V_j=\sum_{k=1}^{K_s}\omega_kP^f_{kj}.
\]

由于 `∑kωk=1`，所以 `∑jVj=1`，便于诊断和跨样本比较。

### 7.5 反向匹配源物体全部前景

每个目标原型不能只与源正热原型比较，而要反向匹配源物体全部前景描述符：

\[
P^b_{ji}=
\frac{\exp((z_j^t)^\top f_i^s/T_b)}
{\sum_{\ell=1}^{N_s}
\exp((z_j^t)^\top f_\ell^s/T_b)}.
\]

将源前景连续热力归一化：

\[
\bar A_s(i)=
\frac{a_i^s}{\sum_{\ell=1}^{N_s}a_\ell^s}.
\]

反向落回任务接触区域的分数：

\[
Q_j=\sum_{i=1}^{N_s}\bar A_s(i)P^b_{ji}.
\]

`Qj` 是两个分布的重合度，不是未经标定的概率。它的绝对尺度会随源热力熵和
源 patch 数变化，因此保存 raw `Q`，但不能直接用固定 `Q>0.5` 做拒绝判断。

### 7.6 循环一致性分数

\[
H_j=V_jQ_j.
\]

把 `Hj` 分配给属于目标 cluster `j` 的所有 foreground patch，得到目标特征网格
热力。随后：

1. 双线性插值到 `518×518` 网络输入；
2. 通过逆 letterbox 映射回目标原图；
3. 乘以 `target_mask`；
4. 只在目标 mask 内做 min-max 归一化；
5. mask 外严格置零；
6. 同时保存归一化前的 `target_heatmap_raw`。

第一版不二值化、不保留 AffCorrs CRF，也不做形态学后处理。这样可以直接观察
循环匹配本身是否成功。

如果目标 mask 内 raw heat 的最大值与最小值之差小于数值阈值，不能强行
min-max 放大噪声；应输出全零归一化热力，并以 `flat_target_heat` 拒绝迁移。

### 7.7 温度参数

AffCorrs 官方实现对原 DINO-ViT 使用约 `T_forward=0.2` 和
`T_backward=0.02`，但 DINOv2、object crop 和连续热力会改变余弦相似度分布，
不能把论文数值当成无需验证的固定常数。

第一版建议默认：

```text
T_forward = 0.10
T_backward = 0.05
```

固定 episode-to-simulation 样例上只做小网格诊断：

```text
T_forward  ∈ {0.05, 0.10, 0.20}
T_backward ∈ {0.02, 0.05, 0.10}
```

每组必须保存相同布局的 `V/Q/H` 可视化，不能只挑主观最好的一张。

## 8. 全局置信度与拒绝迁移

最终热力会被归一化到 `[0,1]`，因此不能用归一化后的最大值作为可信度。置信度
必须从归一化前的匹配量计算。

### 8.1 循环一致性置信度

均匀反向匹配的基线为：

\[
q_{\mathrm{uniform}}=1/N_s.
\]

反向匹配集中到最热源 patch 时的理论上界为：

\[
q_{\mathrm{upper}}=\max_i\bar A_s(i).
\]

若 `q_upper-q_uniform` 接近零，说明源热力在前景上近似均匀，没有可迁移的功能
区域，应以 `flat_source_heat` 拒绝，而不是进入除法。

定义校准后的目标区域循环分数：

\[
\hat Q_j=\mathrm{clip}\left(
\frac{Q_j-q_{\mathrm{uniform}}}
{q_{\mathrm{upper}}-q_{\mathrm{uniform}}+\epsilon},
0,1\right).
\]

使用正向投票作权重：

\[
C_{\mathrm{cycle}}=\sum_jV_j\hat Q_j.
\]

### 8.2 峰值显著度

在目标 mask 内对 raw heat 计算：

\[
C_{\mathrm{peak}}=
\mathrm{clip}\left(
\frac{q_{95}(H)-q_{50}(H)}
{q_{95}(H)+\epsilon},0,1\right).
\]

它用于判断目标热力是否有显著功能区域，而不是在整个杯身近似均匀。

### 8.3 热力熵

考虑 cluster 所覆盖的 patch 数 `nj`：

\[
p_j=\frac{n_jH_j}{\sum_\ell n_\ell H_\ell+\epsilon},
\]

\[
C_{\mathrm{entropy}}=
1-\frac{-\sum_jp_j\log(p_j+\epsilon)}
{\log N_{\mathrm{target\ patch}}}.
\]

分数越高表示结果越集中。该项只是一种启发式质量指标，多处真实接触区域可能
具有较高熵，因此必须在实验中单独报告，不能只保留一个总分。

### 8.4 总分和拒绝

实际第一版把 cycle、peak 和 concentration=`1-entropy` 三项等权做几何平均。
retained-source-heat-mass 单独报告并单独阈值拒绝，不参与 global：

\[
C_{\mathrm{global}}=
\left(
C_{\mathrm{cycle}}\,
C_{\mathrm{peak}}\,
(1-E_{\mathrm{target}})
\right)^{1/3}.
\]

上式中的连乘应在实现中使用 log-domain：

\[
\log C_{\mathrm{global}}=
\frac{1}{3}\sum_{r\in\{\mathrm{cycle,peak,concentration}\}}
\log(\max(C_r,10^{-8})).
\]

实际配置初值：

```text
minimum_global_score: 0.05
minimum_cycle_score: 0.05
minimum_peak_score: 0.05
maximum_entropy: 0.98
minimum_retained_heat: 0.50
```

这些阈值只作为第一版工程拒绝初值，不宣称已经标定。即使被拒绝，也要保存
完整 `npz/json/png` 诊断结果。当前不会把结果自动传给任何点云或抓取模块。

## 9. 当前明确排除的内容

当前版本到 `target_heatmap[Ht,Wt]` 为止，以下内容全部不实现、不测试、不接线：

```text
target_depth / camera_intrinsics
二维像素反投影
target_points / target_contact_score
可见点云热力
完整点云或 CAD 点云
不可见侧/双侧热力传播
antipodal pair
GraspNet / AnyGrasp
top-down 抓取约束
碰撞检测、IK 和机器人执行
Open3D 点云或夹爪可视化
```

完整点云和 GraspNet 仍不是本轮二维方法的一部分，也不作为 Soft Heatmap
AffCorrs 本身的验收依据。它们现已在独立的下游流水线中消费保存后的
`target_heatmap`；这不改变本节列出的二维接口边界。

## 10. 重构后的代码结构

```text
LFV/
├── configs/
│   └── affordance_transfer/
│       ├── soft_heatmap_affcorrs.yaml
│       └── episode0_to_maniskill.yaml
│
├── lfv/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── dinov2_dense.py
│   │
│   ├── affordance_transfer/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── preprocessing.py
│   │   ├── clustering.py
│   │   ├── soft_affcorrs.py
│   │   ├── confidence.py
│   │   ├── adapters.py
│   │   ├── pipeline.py
│   │   ├── io.py
│   │   └── app.py
│   │
│   └── visualization/
│       └── affordance_transfer.py
│
├── scripts/
│   └── affordance_transfer/
│       ├── transfer_contact_heatmap.py
│       └── validate_episode0_to_maniskill.py
│
└── tests/
    ├── test_affordance_transfer_preprocessing.py
    ├── test_weighted_kmeans.py
    ├── test_soft_affcorrs.py
    ├── test_affordance_transfer_confidence.py
    └── test_episode0_maniskill_transfer_smoke.py
```

### 10.1 各模块职责

`lfv/features/base.py`

- 定义冻结稠密特征提取协议；
- 不包含 DINO 专用逻辑。

`lfv/features/dinov2_dense.py`

- 加载本地 DINOv2 权重；
- 执行 ImageNet normalization；
- 提取和 L2 归一化 patch tokens；
- 校验 token 数与 `grid_hw`。

`lfv/affordance_transfer/schema.py`

- 定义 `SourceContactExample`、`TargetObservation`、`CropTransform`、
  `PreparedImage` 和 `TransferResult`；
- 统一 shape/dtype/range 检查。

`preprocessing.py`

- bbox、margin、resize、letterbox；
- RGB/mask/heat 同步变换；
- patch center 与原图的双向坐标映射。

`clustering.py`

- deterministic weighted K-Means++；
- source heat weighting；
- target dense over-clustering；
- 空簇重置，配置簇数超过有效样本数时自动取有效上限。

`soft_affcorrs.py`

- 只实现 `S/Pf/V/Pb/Q/H` 数学；
- 输入已经是归一化描述符和网格值；
- 不读文件、不加载 DINO、不画图。

`confidence.py`

- 计算 cycle、peak、entropy、global；
- 返回 rejection reasons；
- 不修改热力结果。

`adapters.py`

- 从 `episode_0` 读取源 RGB/mask/heat；
- 从 ManiSkill `pouring_snapshot.npz` 只读取目标 RGB/mask；
- 核心算法不依赖 zarr 或 ManiSkill。

`pipeline.py`

- 组合预处理、特征、聚类、匹配、置信度和原图逆映射；
- 暴露稳定接口：

```python
result = pipeline.transfer(source, target)
```

`io.py`

- 保存 `transfer_result.npz` 和 `transfer_report.json`；
- 不包含算法。

`lfv/visualization/affordance_transfer.py`

- 只生成固定的一张 `transfer_summary.png`；
- 不依赖 Open3D；
- 输入为 arrays/result，不负责读取实验配置。

## 11. 第一阶段旧代码的删除与迁移清单

以下删除已在新 smoke test 和真实 DINOv2 验证通过后执行。执行前已创建当前
Stage-1 轻量源码快照；没有删除 `/media/ljian/lj` 下的数据，也没有删除已有
`lfv_runs` 结果。

```text
/home/users1/ljian/LFV_stage1_joint_diffusion_pre_soft_affcorrs_20260731.tar.gz
```

### 11.1 删除：旧 Joint Contact–Grasp 模型闭环

```text
lfv/datasets/contact_grasp.py
lfv/datasets/contact_grasp_schema.py
lfv/diffusion/
lfv/models/contact_grasp_generation/
lfv/models/common/pointnet2.py
lfv/models/registry.py
lfv/geometry/rotation_6d.py
lfv/training/contact_grasp/
lfv/evaluation/contact_grasp/
lfv/visualization/contact_grasp.py
lfv/visualization/contact_grasp_report.py

configs/experiments/contact_grasp/

scripts/preprocess/prepare_contact_grasp_artifacts.py
scripts/train/train_contact_grasp.py
scripts/infer/sample_contact_grasp.py
scripts/infer/run_contact_grasp_qualitative_suite.py
scripts/evaluate/evaluate_contact_grasp.py
scripts/evaluate/summarize_contact_grasp_training.py
scripts/evaluate/audit_grasp_label_geometry.py
scripts/evaluate/render_grasp_label_audit_cases.py
scripts/visualize/visualize_contact_grasp.py
scripts/visualize/render_contact_grasp_open3d.py
scripts/visualize/calibrate_open3d_view.py
```

相应测试删除：

```text
tests/test_contact_grasp_checkpoint.py
tests/test_contact_grasp_data.py
tests/test_contact_grasp_open3d_geometry.py
tests/test_contact_grasp_overfit.py
tests/test_contact_grasp_report.py
tests/test_diffusion_schedulers.py
tests/test_joint_contact_grasp_model.py
tests/test_rotation_6d.py
tests/test_grasp_label_audit.py
```

### 11.2 删除或退出活动流程：旧抓取伪标签分支

新 Stage 1 不再从单视角 HaMeR 关键点构造训练用 grasp pose，因此以下代码退出
主流程，在确认没有其他任务依赖后删除：

```text
lfv/pipeline/hamer_hand_pose.py
lfv/pipeline/thumb_index_grasp_label.py
scripts/run_hand_pouring_grasp_batch.sh
tools/process_episode0_hamer_thumb_index_grasp.py
tools/visualize_hamer_thumb_index_grasp_open3d.py
tools/visualize_episode0_hamer_thumb_index_grasp_open3d.py
tools/check_hand_pouring_grasp_batch.py
tools/verify_episode0_graspnet_contact_roi.py
scripts/visualize_episode0_graspnet_contact_roi.sh
```

已有 episode 中的 HaMeR/graspnet 验证 artifact 不在代码重构中删除，只是不再被
新 pipeline 消费。

### 11.3 当前里程碑中的替换

```text
scripts/sim/predict_and_propagate_pouring_contact.py
    ->
scripts/affordance_transfer/transfer_contact_heatmap.py
scripts/affordance_transfer/validate_episode0_to_maniskill.py

configs/experiments/contact_grasp/*
    ->
configs/affordance_transfer/soft_heatmap_affcorrs.yaml
configs/affordance_transfer/episode0_to_maniskill.yaml
```

二维验证命令仍独立运行到 `target_heatmap` 为止。旧 learned-contact 入口在删除
Joint Diffusion 时一并退出活动流程；新的可选下游使用
`scripts/sim/run_transferred_heat_topdown_grasp.py`，只读取二维落盘结果，不修改
`soft_affcorrs.py` 或二维输出 schema。

### 11.4 保留

```text
lfv/pipeline/contact_heatmap.py
lfv/pipeline/contact_timing.py
lfv/pipeline/sam2_mask.py
lfv/pipeline/dinov2_features.py
lfv/data_processing/

lfv/geometry/contact_heat_propagation.py
lfv/geometry/oracle_contact.py

lfv_sim/maniskill/
scripts/sim/export_pouring_contact_snapshot.py
scripts/sim/create_oracle_handle_contact.py
scripts/sim/generate_graspnet_from_full_contact.py
scripts/sim/view_pouring_contact_grasp_open3d.py
scripts/sim/render_pouring_contact_camera_view.py

tests/test_contact_field_core.py
tests/test_contact_heat_propagation.py
tests/test_oracle_handle_contact.py
```

`lfv/pipeline/dinov2_features.py` 保留离线 episode 功能，其本地权重分支已改为
调用新的 `lfv/features/dinov2_dense.py`；仅为兼容旧配置保留 Hugging Face
fallback。

依赖已删除 learned contact predictor 的
`tests/test_pouring_contact_grasp_pipeline.py` 与集成调度器同步移除。独立的
oracle/完整表面工具保留，但当前二维迁移不会导入或调用它们。

### 11.5 依赖清理

删除旧模型后，如全仓库无引用，从 `requirements.txt` 移除：

```text
diffusers
tensorboard
wandb
accelerate
```

继续保留：

```text
torch
timm
opencv-python
numpy
 scipy
 zarr
 trimesh
```

Open3D 已从核心 `requirements.txt` 移除。下游完整点云/GraspNet 流水线使用独立
`graspnet` 环境运行 Open3D/Xvfb，不构成当前二维迁移依赖。

## 12. 文档清理记录

已删除失效的 Joint Diffusion、单视角 grasp label 和旧集成抓取文档：

```text
docs/joint_contact_grasp_diffusion_v1_zh.md
docs/grasp_label_geometry_audit_zh.md
docs/hamer_grasp_pseudo_label_plan.md
docs/hand_pouring_contact_and_grasp_handoff.md
docs/pouring_complete_contact_grasp_validation_zh.md
docs/repository_reorganization_for_two_stage_generation.md
docs/new_task_data_processing_runbook.md
```

已重写或更新：

```text
README.md
docs/project_architecture_and_development_guide_zh.md
scripts/*/README.md
lfv/*/README.md
```

保留：

```text
docs/hand_pouring_dino_sam_processing.md
docs/contact_field_data_processing_plan.md
```

本文件在实现完成后从“计划”更新为“方法、接口与运行说明”，记录真实参数、
真实输出和失败案例，而不是另建一份重复文档。

## 13. 测试设计

### 13.1 预处理与映射

`test_affordance_transfer_preprocessing.py`

- 非方形 bbox 经过同一 RGB/mask/heat letterbox；
- 任意原图坐标的 original → input → original round trip；
- feature grid 逆映射后 shape 正确且 mask 外严格为零；
- 非 patch-size 整数倍输入被拒绝。

### 13.2 Weighted K-Means

`test_weighted_kmeans.py`

- 两个合成 descriptor 簇被分离；
- cluster mass 总和等于输入权重总和；
- 固定 seed 完全复现 labels/centroids。

### 13.3 Soft AffCorrs 数学

`test_soft_affcorrs.py`

用两个正交语义区域构造 toy descriptors，同时检查：

```text
H_semantic_match > 20 * H_unrelated
Pf 每一行和为 1
Pb 每一行和为 1
source omega 和为 1
source 只在 heat > tau 的 patch 上聚类
配置 K 超过正样本数时取有效上限
```

### 13.4 置信度与拒绝

`test_affordance_transfer_confidence.py`

- 输出必须含 `global/cycle/peak/entropy`；
- 人为提高 global 阈值后必须拒绝并返回
  `low_global_confidence`。

### 13.5 CPU 集成测试

使用确定性的 RGB/坐标假特征 extractor，不加载 DINO 权重，运行：

```text
preprocess metadata
-> clustering
-> soft matching
-> confidence
-> inverse mapping
```

检查 target shape、dtype、mask 外严格为零、目标峰值位于 mask 内。另构造含
pickle-object depth/point 字段的 NPZ，证明 target adapter 不会访问这些字段。

### 13.6 GPU smoke test

实际加载 DINOv2：

```text
episode_0 frame 39
    ->
ManiSkill seed_0_dataset_aligned RGB image
```

固定输出目录：

```text
/home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/
episode_0_to_maniskill_seed_0/
```

硬性结构验收：

- `target_heatmap.shape == [480,640]`；
- mask 外严格为 0；
- target heat 在 `[0,1]` 且保留连续值；
- 相同 seed 重跑的 `npz` 数值一致；
- 报告包含 source/target crop 映射、特征模型和全部置信度。

语义验收不只看归一化最大值：

- 峰值位于杯子 mask 内；
- 预测高热主要集中在目标把手，而不是杯沿或杯身；
- 保存 peak UV 和 centroid UV，人工核对固定 PNG；
- 当前没有标注独立的 target handle GT mask，因此不虚构 Soft-IoU 或
  peak-to-handle 数值。建立多实例标注集后再增加这类客观指标。

## 14. 固定的一张可视化

主测试只要求打开一张：

```text
transfer_summary.png
```

推荐固定为 `2×3`：

```text
┌────────────────┬────────────────┬────────────────┐
│ source RGB     │ source heat    │ source positive│
│ + object mask  │ overlay        │ patches/protos │
├────────────────┼────────────────┼────────────────┤
│ target RGB     │ target heat    │ V / Q / H      │
│ + object mask  │ overlay        │ diagnostics    │
└────────────────┴────────────────┴────────────────┘
```

图中必须标注：

```text
source/target sample id
K_source / K_target
T_forward / T_backward
cycle / peak / entropy / global confidence
accepted 或 rejection reasons
目标高热峰值像素
```

颜色范围固定 `[0,1]`，source 和 target 使用同一 colormap。不能对每张图自动改变
色条后只凭颜色判断效果。诊断面板分别显示映射回目标 patch 的 `V`、校准前 `Q`
和最终 `H`，这样能够区分：

- DINO 正向匹配错；
- 反向循环验证拒绝；
- 两者都正确但目标热力过于分散。

当前只生成这一张二维 PNG，不生成 Open3D、点云或夹爪视图。报告额外保存
源/目标热力的 peak UV、加权 centroid UV 和 heat mass，便于后续批量回归检查
峰值是否发生漂移。

## 15. 配置结构

```yaml
seed: 0

source:
  episode_dir: /media/ljian/lj/data_3d/hand_pouring_lfv/episode_0
  frame_index: 39
  mask_path: sam_mask/affordance_mask.npy
  heatmap_path: contact_heatmap/contact_heatmap.npz
  heatmap_key: heatmap_2d

target:
  snapshot_path: /home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/seed_0_dataset_aligned/pouring_snapshot.npz
  rgb_key: rgb
  mask_key: cup_mask

features:
  backend: dinov2
  model_name: vit_small_patch14_dinov2
  weights_path: /home/users1/ljian/LFV/third_party/dinov2_weights/dinov2_vits14_pretrain.pth

preprocessing:
  input_size: 518
  bbox_margin: 0.15
  mask_occupancy_threshold: 0.35

matching:
  source_clusters: 6
  target_clusters: 64
  positive_threshold: 0.20
  forward_temperature: 0.10
  backward_temperature: 0.05
  n_init: 4
  max_iter: 100
  seed: 0

confidence:
  minimum_retained_heat: 0.50
  minimum_cycle_score: 0.05
  minimum_peak_score: 0.05
  maximum_entropy: 0.98
  minimum_global_score: 0.05

runtime:
  device: cuda

output:
  directory: /home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/episode_0_to_maniskill_seed_0
```

## 16. 实施顺序

### Phase A：冻结旧结果并建立新核心

1. 生成待删除文件 manifest；
2. 在活动仓库外保存当前未提交 Stage-1 轻量源码快照；
3. 新建 schema、preprocessing、weighted K-Means 和纯数学测试；
4. 不连接 DINO 和真实数据，先通过 toy cycle test。

### Phase B：接入 DINOv2 和固定样例

1. 抽取通用 DINOv2 backend；
2. 让旧 episode DINO pipeline 调用同一 backend；
3. 接入 episode_0 source adapter；
4. 接入 ManiSkill snapshot target adapter；
5. 输出单张固定诊断图和 `npz/json`；
6. 执行温度小网格并保留所有结果。

### Phase C：删除旧 Stage-1 并更新文档

1. 新 GPU smoke test 和全部单元测试通过；
2. 按清单删除 Joint Diffusion、训练、数据集、旧测试和旧训练可视化；
3. 清理 imports、requirements、README 和旧文档；
4. 全仓库执行 `rg`，确保没有旧 checkpoint/config 路径；
5. 运行完整 pytest 和固定二维 transfer；
6. 将本文件由计划更新为真实实现说明。

### Phase D：后续多源扩展

稳定后允许：

```text
同一演示的多个清晰源帧
    -> 每帧独立执行完整 Soft AffCorrs
    -> 每帧得到 H_t^(m) 和 C_cycle^(m)
    -> 按置信度归一化融合
```

\[
\alpha_m=
\frac{C_{\mathrm{cycle}}^{(m)}}
{\sum_rC_{\mathrm{cycle}}^{(r)}+\epsilon},
\qquad
H_{\mathrm{fused}}=\sum_m\alpha_mH_t^{(m)}.
\]

多帧之间还应计算峰值位置和热力分布的一致性。第一版不要提前实现该分支。

## 17. 完成定义

只有同时满足以下条件，才能认为本轮重构完成：

1. 活动仓库中不再存在旧 Joint Contact–Grasp 训练和采样路径；
2. 源 episode 连续热力可通过一个命令迁移到 ManiSkill 目标图；
3. 输出原始和归一化后的二维连续目标热力；
4. 一张固定 PNG 能同时判断源输入、目标结果和 `V/Q/H` 失败位置；
5. 置信度和拒绝原因可复现，并保存归一化前诊断量；
6. target mask、bbox/letterbox 和原图逆映射有单元测试；
7. 固定 seed 的 DINO/聚类结果可复现；
8. README 和项目结构文档不再把旧 checkpoint 当作当前 Stage-1；
9. 当前命令完全不读取 depth/intrinsics，也不调用点云或 GraspNet 代码；
10. 旧代码只存在于仓库外备份，不再污染 LFV 活动结构。

## 18. 2026-07-31 实施与验证记录

### 18.1 已实施内容

- 建立 `DenseFeatureExtractor` 协议和本地权重
  `DinoV2DenseExtractor`；
- 将旧 episode DINO 特征 stage 的本地权重分支复用到同一 extractor；
- 实现同步 bbox/letterbox、mask/heat 网格化与原图逆映射；
- 实现确定性加权 K-Means++；
- 实现 `Pf/V/Pb/Q/H` 纯数学核心；
- 实现 cycle、peak、entropy、retained heat 和 global confidence；
- 实现 source episode 与 target snapshot adapter；
- 实现 `transfer_result.npz`、`transfer_report.json` 和固定六联图；
- 实现通用 CLI 与拒绝时非零退出的固定验证 CLI；
- 删除旧 Joint Diffusion、训练、采样、旧测试、旧 Open3D 训练可视化和
  单视角 HaMeR 抓取伪标签活动链路；
- 重写 README 与项目架构文档。

当前方法是 training-free，因此这轮没有“训练 checkpoint”；冻结 DINOv2 只做
前向特征提取，所有可迭代参数均在 YAML 中。

### 18.2 测试结果

```text
python -m compileall -q lfv scripts tools tests
python -m pytest -q

14 passed
git diff --check: passed
```

其中二维迁移新增测试为 8 个，其余 6 个是保留的源 contact-field 与独立几何
工具回归。目标 adapter 的测试在 NPZ 中放入需要 pickle 才能读取的 depth/point
object 字段，确认当前代码只访问 RGB 和 mask。

### 18.3 固定真实运行

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/validate_episode0_to_maniskill.py \
  --config configs/affordance_transfer/episode0_to_maniskill.yaml
```

实际运行：

```text
device: cuda
feature grid: [37, 37, 384]
source foreground patches: 483
source heat-positive patches: 30
source prototypes: 6
target foreground patches: 607
target regions: 64

accepted: true
global: 0.3365783966
cycle: 0.0986442566
peak: 0.9998947045
entropy: 0.6134260019
retained_heat_mass: 0.8975862876
target peak UV: (208, 333)
target centroid UV: (209.601, 324.100)
```

人工检查固定 PNG：目标高热集中在仿真杯子的左侧把手，没有迁移到杯身或右侧
碗。V、Q 和 H 都把最高响应放在把手局部。

### 18.4 确定性

在相同 GPU、配置和本地权重上连续运行两次，以下数组的 SHA-256 完全一致：

```text
target_heatmap
ea6a081f9ce3765ea00923215268e9fd2b4729526475f32d5f279aa59acbe64c

target_heatmap_raw
582bdb4508ce0c2a95d2644edc14a59d919f89fb5fa081ad61e22ed03029de15

cycle_score_grid
8118e987a4f678500481264b6f5f3c12958b49652f01d2c00ba4fe7f766cb8fb
```

### 18.5 固定产物

```text
/home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/
└── episode_0_to_maniskill_seed_0/
    ├── transfer_result.npz
    ├── transfer_report.json
    └── transfer_summary.png
```

源码清理前备份：

```text
/home/users1/ljian/LFV_stage1_joint_diffusion_pre_soft_affcorrs_20260731.tar.gz
```
