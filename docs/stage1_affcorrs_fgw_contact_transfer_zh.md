# LFV Stage 1 完成版：Soft Heatmap AffCorrs + FGW Contact Field Transfer

更新日期：2026-08-04
状态：Stage 1 方法、代码、配置、测试、双任务真实验证和固定可视化均已完成。

## 1. Stage 1 要解决的问题

Stage 1 的任务不是直接预测夹爪位姿，也不是学习机器人运动轨迹，而是把人类演示
中的任务接触知识迁移到一个同类别新实例：

```text
源人类演示：source RGB-D + 完整功能部件 mask + 连续 Contact Field
                                  │
                                  ▼
                      Stage 1 one-shot transfer
                                  │
                                  ▼
目标观测：target RGB-D + 完整功能部件 mask + 连续 Contact Field
```

输出的目标 Contact Field 保持连续概率，不二值化，保存于目标相机像素坐标系中，
可以被后续的完整表面传播和 GraspNet 消费。

本阶段允许使用源、目标**可见 RGB-D**建立功能部件内部结构对应，但明确不使用：

- 仿真器完整 mesh 或不可见表面点云；
- 单视角热力向隐藏侧的传播；
- GraspNet 或任何夹爪候选；
- 机器人 IK、闭合、运动轨迹或执行结果。

因此，Stage 1 的成功标准是“目标实例的正确功能部件以及部件内部正确位置获得
连续热力”，不是抓取或任务执行成功率。抓取只作为保存结果的下游可执行性验证。

## 2. 总体方法

完整计算由两个互补模块组成：

```text
                    source RGB + heat             target RGB
                           │                           │
                           └──── frozen DINOv2 ───────┘
                                         │
                              Soft Heatmap AffCorrs
                                         │
                        目标中的语义功能区域 A_aff
                                         │
             ┌───────────────────────────┴────────────────────────────┐
             │                                                        │
source full-part RGB-D                                      target full-part RGB-D
             │                                                        │
             └──── point DINO + FPS + kNN geodesic + balanced FGW ──┘
                                         │
                        部件内结构 Contact Field A_fgw
                                         │
                    AffCorrs semantic gate × FGW structural field
                                         │
                                         ▼
                            final target Contact Field
```

两部分的职责严格区分：

- **AffCorrs**：回答“目标中的哪个语义区域与源任务接触部件对应”；
- **FGW**：回答“源 Contact Field 在整个部件内部的结构位置应对应到目标哪里”。

## 3. 原版 AffCorrs 做了什么

本项目参考 _One-Shot Transfer of Affordance Regions? AffCorrs!_ 的核心骨架。为
避免把 LFV 的扩展错误归因于原论文，先单独说明原方法。

设源图中有一个二值查询区域 `query mask`，目标图中需要找到与它语义对应的
affordance region。原版 AffCorrs 的主要计算是：

1. 用冻结 DINO-ViT 提取源、目标稠密 patch 描述符；
2. 只对源二值查询区域中的描述符聚类，形成少量 query prototypes；
3. 对目标显著前景进行较密集的区域聚类；官方实现还处理目标背景 clusters；
4. 每个源 query prototype 正向匹配目标区域 prototypes，并对目标区域投票；
5. 每个目标区域 prototype 反向匹配源图全部描述符；
6. 计算反向概率有多少重新落入源二值 query mask；
7. 将正向投票和反向落回概率相乘，形成循环一致性区域分数；
8. 把区域分数恢复到像素并经 CRF 得到二值目标区域。

这里最重要的不是单向最近邻，而是“源查询区域 → 目标区域 → 源完整图像”的
双向验证。目标中的一个区域即使和源查询区域有一定相似度，如果反向更容易落到
源物体的其他部位，其循环分数也会被抑制。

原版 AffCorrs 主要解决的是**语义区域对应**，并不显式约束源、目标功能部件内部
点对之间的几何距离关系。因此它可以判断“这是把手”，但不能稳定表达“这是把手
中央而不是两端”。

## 4. LFV 对 AffCorrs 的第一层改造：Soft Heatmap AffCorrs

### 4.1 输入和统一预处理

源输入：

```text
source_rgb       uint8   [Hs,Ws,3]
source_mask      bool    [Hs,Ws]
source_heatmap   float32 [Hs,Ws], [0,1]
```

目标输入：

```text
target_rgb       uint8   [Ht,Wt,3]
target_mask      bool    [Ht,Wt]
```

源和目标分别从 mask 求紧致 bbox，按相同比例增加 margin，等比例缩放并 letterbox
到 `518×518`。RGB 使用双线性插值，mask 使用最近邻/占比判定，连续热力使用双
线性插值。`CropTransform` 保存原图、crop、resize、padding 和 DINO 输入之间的
双向坐标映射。

冻结 DINOv2 ViT-S/14 输出：

```text
F_s_grid, F_t_grid: [37,37,384]
```

每个 patch descriptor 逐向量 L2 归一化。只保留 mask 内且不属于 padding 的
前景 patch，得到：

\[
F_s=\{f_i^s\}_{i=1}^{N_s},\qquad
F_t=\{f_j^t\}_{j=1}^{N_t},\qquad
A_s=\{a_i^s\}_{i=1}^{N_s}.
\]

### 4.2 连续正热集合

原 AffCorrs 使用二值 query mask。LFV 保留完整连续热力，同时只用超过阈值的
patch 建立源接触 prototypes：

\[
\mathcal P=\{i\mid a_i^s>\tau_{pos}\}.
\]

记录阈值保留的源热力质量：

\[
r_{keep}=\frac{\sum_{i\in\mathcal P}a_i^s}{\sum_i a_i^s}.
\]

当前固定配置为 `tau_pos=0.20`。

### 4.3 热力加权源 K-Means

对正热 patch 做加权 K-Means：

\[
\min_{z_k^s,c_i}
\sum_{i\in\mathcal P}a_i^s\lVert f_i^s-z_{c_i}^s\rVert_2^2.
\]

cluster center 在每轮更新后重新 L2 归一化。第 `k` 个源 prototype 的热力质量：

\[
m_k=\sum_{i\in\mathcal P,c_i=k}a_i^s,
\qquad
\omega_k=\frac{m_k}{\sum_r m_r}.
\]

当前使用 `K_s=6`。这一步和原版二值 query 聚类的区别是：源热力越高的 patch，
越影响 prototype 位置和后续投票权重。

### 4.4 目标功能区域过聚类

在目标 mask 内对所有前景 DINO 特征做普通 K-Means：

\[
\min_{z_j^t,d_l}\sum_l\lVert f_l^t-z_{d_l}^t\rVert_2^2.
\]

当前固定 `K_t=64`。这里的 64 是 AffCorrs 目标区域 cluster 数，和后面 FGW 的
点云节点数没有关系。

### 4.5 正向热力投票

计算源接触 prototypes 与目标区域 prototypes 的余弦相似度：

\[
S_{kj}=\langle z_k^s,z_j^t\rangle.
\]

对每个源 prototype 沿所有目标区域做温度 Softmax：

\[
P^f_{kj}=
\frac{\exp(S_{kj}/\tau_f)}
{\sum_r\exp(S_{kr}/\tau_f)}.
\]

再用源 prototype 的热力质量投票：

\[
V_j=\sum_k\omega_kP^f_{kj}.
\]

`V_j` 高表示多个高热源 prototypes 都认为目标第 `j` 个区域是合理对应区域。
当前 `tau_f=0.10`。

### 4.6 反向连续热力验证

对每个目标区域 prototype，与**源 mask 内全部前景描述符**计算相似度：

\[
B_{ji}=\langle z_j^t,f_i^s\rangle.
\]

沿源全部位置做温度 Softmax：

\[
P^b_{ji}=
\frac{\exp(B_{ji}/\tau_b)}
{\sum_r\exp(B_{jr}/\tau_b)}.
\]

将源连续热力归一化为概率分布：

\[
\bar A_s(i)=\frac{a_i^s}{\sum_r a_r^s}.
\]

目标区域反向落回源 Contact Field 的概率为：

\[
Q_j=\sum_iP^b_{ji}\bar A_s(i).
\]

原版使用二值 query mask 累积概率；LFV 用连续热力加权，因此反向落到源峰值比
落到源低热边缘获得更高分。当前 `tau_b=0.05`。

### 4.7 循环一致性区域分数

最终 Soft Heatmap AffCorrs 区域分数：

\[
H_j^{aff}=V_jQ_j.
\]

把 `H_j^{aff}` 分配给目标 cluster `j` 中的所有 patch，经保存的逆映射插值回
目标原图，乘以目标 mask，得到：

\[
A_{aff}(u,v)\in[0,1].
\]

LFV 不使用 CRF，也不将结果二值化。到这里，系统已经能判断目标中的正确语义
部件，但仍可能把源把手中央的 Contact 扩散成整个目标把手。

### 4.8 迁移置信度

先定义源均匀反向基线和理论上限：

\[
q_0=1/N_s,\qquad q_{max}=\max_i\bar A_s(i).
\]

校准目标区域的反向分数：

\[
\tilde Q_j=
\operatorname{clip}\left(\frac{Q_j-q_0}{q_{max}-q_0},0,1\right).
\]

循环一致性置信度：

\[
c_{cycle}=\frac{\sum_jV_j\tilde Q_j}{\sum_jV_j}.
\]

目标峰值显著度使用正分数的 50% 与 95% 分位数：

\[
c_{peak}=\operatorname{clip}
\left(\frac{q_{95}-q_{50}}{q_{95}},0,1\right).
\]

将目标 patch 分数归一化为分布 `p_j`，计算归一化熵：

\[
e=-\frac{\sum_jp_j\log p_j}{\log N_t},\qquad
c_{conc}=1-e.
\]

全局置信度为三者几何平均：

\[
c_{global}=(c_{cycle}c_{peak}c_{conc})^{1/3}.
\]

`retained_heat_mass` 单独设门限，不用于抬高全局分数。任一门限不满足时，输出会
记录明确 rejection reason。

## 5. LFV 对 AffCorrs 的第二层改造：FGW 部件内结构迁移

### 5.1 为什么 AffCorrs 后还需要 FGW

AffCorrs 的目标 cluster 由 DINO 语义划分，并不知道 cluster 在整个把手的中央、
端部或连接处。增加 `K_t` 可以让区域更细，但不能从目标函数上保证部件内部空间
关系，所以不能根治“整段把手都亮”的问题。

FGW 同时优化跨对象语义相似和对象内部成对距离关系，使匹配满足：

- 对应点的 DINO 语义应该相近；
- 如果两个源点在部件结构中相距较近/较远，其目标对应点也应保持类似关系。

### 5.2 RGB-D 反投影与逐点 DINO

FGW 输入必须是完整可见功能部件 mask，而不是源高热区。对 mask 内合法深度像素：

\[
z=D(u,v),\quad
x=(u-c_x)z/f_x,\quad
y=(v-c_y)z/f_y.
\]

得到 OpenCV camera frame 中的三维点：

\[
p=[x,y,z]^T,
\]

其中 `+x` 向右、`+y` 向下、`+z` 向前。

利用 `CropTransform.original_to_input()` 把原图像素映射到 DINO 输入，再按 patch
中心坐标双线性采样 DINO 网格，得到逐点特征：

\[
(p_i^s,f_i^s,h_i^s),\qquad(p_j^t,f_j^t).
\]

源热力 `h_i^s` 直接来自同一像素的连续 Contact Field。RGB、深度、mask、热力和
DINO 坐标通过显式映射对齐，不允许独立随机采样。

### 5.3 FPS 下采样

源、目标完整可见部件点云分别执行确定性 Euclidean FPS：

```text
source FGW nodes: Ns ≈ 256
target FGW nodes: Nt ≈ 256
```

当前固定使用 256 个节点。这里的 256 是结构图节点数，不是 AffCorrs 的 `K_t=64`。

### 5.4 跨对象语义代价

逐点 DINO 特征 L2 归一化后，构造：

\[
M_{ij}=\frac{1-\langle f_i^s,f_j^t\rangle}{2}\in[0,1].
\]

`M_ij` 越小，源节点 `i` 和目标节点 `j` 的视觉语义越相似。

第一版没有把 AffCorrs cluster 分数伪装成不存在的逐点 pairwise cycle cost；
AffCorrs 通过后面的语义门控发挥作用。未来只有在定义并验证真正的点对循环矩阵
后，才能把它加入 `M_ij`。

### 5.5 部件内部测地距离

源、目标点云分别建立 kNN 图。边权为三维欧氏长度，只保留不超过以下阈值的局部
边：

\[
d_{edge}\le r_e\operatorname{median}(d_{NN}).
\]

当前 `k=10`、`r_e=4.0`；若图不连通，`k` 逐步增加到配置上限，仍不连通则明确
失败，不静默用无限距离继续计算。

在两个图上分别计算全对最短路径：

\[
G_s(i,k)=d_{graph}(p_i^s,p_k^s),\qquad
G_t(j,l)=d_{graph}(p_j^t,p_l^t).
\]

每一侧分别用非零测地距离的 95% 分位数归一化并截断：

\[
D_s=\operatorname{clip}(G_s/q_{0.95}(G_s),0,1),
\]

\[
D_t=\operatorname{clip}(G_t/q_{0.95}(G_t),0,1).
\]

这样结构项不依赖源/目标相机的平移、旋转和绝对尺度。

### 5.6 Balanced Fused Gromov-Wasserstein

定义均匀边缘质量：

\[
a_i=1/N_s,\qquad b_j=1/N_t.
\]

传输矩阵满足：

\[
T\mathbf 1=a,\qquad T^T\mathbf 1=b,\qquad T_{ij}\ge0.
\]

用平方结构损失求解：

\[
\min_T
(1-\alpha)\sum_{ij}M_{ij}T_{ij}
+\alpha\sum_{ikjl}
\left(D_s(i,k)-D_t(j,l)\right)^2T_{ij}T_{kl}.
\]

- 第一项要求对应点的 DINO 语义相似；
- 第二项要求源/目标部件内部的相对距离关系一致；
- `alpha=0` 退化为纯语义 OT；
- `alpha=1` 只关注结构，不使用跨对象外观语义；
- 当前第一版使用 `alpha=0.5`。

正式实现调用 POT 的 non-entropic balanced
`fused_gromov_wasserstein`。如果 POT 不可导入且两侧节点数相等，代码提供确定性
SciPy Frank-Wolfe 兜底；正式蓝杯与抽屉结果均使用 POT 0.9.7.post1。

### 5.7 直接输运源 Contact Field

FGW 的作用不是重新分割目标，而是直接把源概率场通过 `T` 输运：

\[
h_j^{fgw}=
\frac{\sum_iT_{ij}h_i^s}
{\sum_iT_{ij}+\epsilon}.
\]

这里除以目标节点接收的传输质量，所以得到的是条件期望意义下的目标 Contact
概率，而不是被节点数量缩小的总质量。

再用目标 FGW 节点的三个最近邻，对目标完整可见部件点云做反距离插值：

\[
h^t(p)=\frac{\sum_{j\in\mathcal N_3(p)}w_j(p)h_j^{fgw}}
{\sum_{j\in\mathcal N_3(p)}w_j(p)},
\qquad
w_j(p)=\frac1{\lVert p-p_j^t\rVert+\epsilon}.
\]

最后按保存的像素索引恢复为目标图像 `A_fgw(u,v)`。

### 5.8 AffCorrs 语义门控与最终输出

如果输入 mask 已经准确限制为完整功能部件，例如 drawer 的完整黑色把手，则
FGW 输出可以直接使用。若旧数据只有整物体 mask，例如 pouring 杯子，则用
AffCorrs 结果抑制杯身上的结构误匹配：

\[
G(u,v)=A_{aff}(u,v)^\gamma,
\]

\[
A_{final}(u,v)=A_{fgw}(u,v)
\left[\lambda+(1-\lambda)G(u,v)\right]M_t(u,v).
\]

- drawer：`lambda=1.0`，等于不使用门控；
- 蓝杯：`lambda=0.05`、`gamma=0.5`，保留少量 FGW floor，同时抑制杯身响应。

`A_final` 不做样本内 min-max，直接保存真实概率值。只有诊断 PNG 为了看清空间
分布，单独使用显示归一化。

## 6. 与原版 AffCorrs 的逐项对比

| 计算环节 | 原版 AffCorrs | LFV Stage 1 |
|---|---|---|
| 视觉特征 | DINO-ViT | 冻结 DINOv2 ViT-S/14 |
| 源查询 | 二值 query mask | 连续 Contact Field + 正热阈值 |
| 源聚类 | query descriptors 聚类 | 热力加权 K-Means |
| 源投票权重 | query prototypes/区域投票 | prototype 热力质量 `omega_k` |
| 反向验证 | 落回二值 query mask | 落回源连续热力分布 |
| 目标输出 | 区域分数 + CRF 二值区域 | 连续 AffCorrs heat，不用 CRF |
| 目标前景 | 显著前景/背景处理 | 显式目标功能部件 mask |
| 坐标映射 | 原实现预处理 | bbox+letterbox+显式可逆映射 |
| 部件内部结构 | 无显式约束 | RGB-D kNN geodesic + FGW |
| 源场迁移 | 不直接输运连续场 | `T` 的条件期望输运 |
| 失败处理 | 主要输出区域 | cycle/peak/entropy 置信度与拒绝原因 |
| 最终表示 | 二值 affordance region | 下游可消费的连续 Contact Field |

因此，LFV 不是简单“在 AffCorrs 后平滑热力”。它先把原版二值语义区域对应改成
连续概率对应，再新增一个独立的三维结构最优传输问题来恢复部件内部位置。

## 7. 固定接口和保存格式

配置分发：

```yaml
method: soft_heatmap_affcorrs  # 纯二维基线
# 或
method: affcorrs_fgw           # Stage 1 完成版
```

Stage 1 完成版输出：

```text
transfer_result.npz
  target_heatmap                 [Ht,Wt] 最终连续 Contact Field
  target_heatmap_raw             [Ht,Wt] 未门控 FGW field
  affcorrs_target_heatmap        [Ht,Wt] AffCorrs K=64 基线
  target_heatmap_fgw_raw         [Ht,Wt]
  semantic_gate                  [Ht,Wt]
  transport                      [Ns,Nt]
  semantic_cost                  [Ns,Nt]
  source_geodesic                [Ns,Ns]
  target_geodesic                [Nt,Nt]
  source/target node points, pixels and heat

transfer_report.json
  schema_version=2
  method、配置、输入来源、solver、图连通统计、置信度和拒绝原因

transfer_summary.png
  固定 2×4 源热力、FGW 节点、语义代价、AffCorrs/FGW/final A/B 图

transfer_source_target_2x2.png
  简洁源/目标快速检查图
```

Stage 1 报告明确声明：

```json
{
  "uses_target_rgb": true,
  "uses_target_mask": true,
  "uses_target_depth": true,
  "uses_point_cloud": true,
  "uses_graspnet": false
}
```

这里的 `uses_point_cloud` 只指可见 RGB-D 部件点云，不代表使用完整 mesh。

## 8. 代码结构

```text
lfv/features/
  base.py                         可替换稠密特征协议
  dinov2_dense.py                 冻结 DINOv2 patch token

lfv/affordance_transfer/
  schema.py                       2D 输入、RGBDPart、TransferResult
  adapters.py                     episode/snapshot 数据适配
  preprocessing.py                crop/letterbox/坐标映射
  clustering.py                   确定性加权 K-Means
  soft_affcorrs.py                V、Q、H 连续循环匹配
  confidence.py                   置信度和拒绝迁移
  pipeline.py                     纯 Soft Heatmap AffCorrs
  fgw_contact_transfer.py         RGB-D、FPS、geodesic、FGW、field transport
  app.py                          method registry/配置分发
  io.py                           NPZ/JSON schema

lfv/visualization/
  affordance_transfer.py          固定 Stage 1 A/B 图

scripts/affordance_transfer/
  transfer_contact_heatmap.py     统一入口

tests/
  test_affordance_transfer_preprocessing.py
  test_weighted_kmeans.py
  test_soft_affcorrs.py
  test_affordance_transfer_confidence.py
  test_fgw_contact_transfer.py
```

## 9. 固定真实配置与结果

### 9.1 Drawer handle

配置：

```text
configs/affordance_transfer/
drawer_episode60_handle_only_fgw_k64_to_maniskill_front.yaml
```

输入使用 source/target 完整黑色把手 mask：

```text
valid RGB-D points                  840 / 1278
AffCorrs target clusters            64
FGW nodes                           256 / 256
alpha                               0.5
FGW objective                       0.121223
AffCorrs > 0.5 peak support         791 pixels
FGW > 0.5 peak support              620 pixels
```

结果把原本较分散的把手响应收回到把手中央连续区域。

### 9.2 Cole blue mug

配置：

```text
configs/affordance_transfer/episode0_to_cole_blue_mug_fgw_k64.yaml
```

```text
valid RGB-D points                  2937 / 10175
AffCorrs target clusters            64
FGW nodes                           256 / 256
alpha                               0.5
FGW objective                       0.123663
AffCorrs > 0.5 peak support         228 pixels
final FGW > 0.5 peak support        23 pixels
AffCorrs global confidence          0.439929
```

最终高热落在蓝色杯子左侧把手。该旧 pouring 数据只有整杯 mask，没有严格的
handle-only mask，因此它验证的是“整杯结构 FGW + AffCorrs 软门控”的工程闭环，
不是最纯净的完整把手到完整把手 FGW 实验。

两个 Stage 1 结果均被后续 GraspNet 成功消费并产生零严格碰撞 IoU 的跨两侧抓取；
这只作为可执行性检查，不属于 Stage 1 优化目标。

## 10. 运行与测试

抽屉：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/drawer_episode60_handle_only_fgw_k64_to_maniskill_front.yaml
```

蓝杯：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/episode0_to_cole_blue_mug_fgw_k64.yaml
```

测试：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python -m pytest -q
```

截至 2026-08-04，仓库测试结果为 `37 passed`。Stage 1 专项覆盖：

- crop、letterbox、mask/heat 对齐和坐标 round-trip；
- 热力加权 K-Means 和确定性；
- AffCorrs 正向/反向 Softmax、`V/Q/H`；
- confidence 和 rejection reason；
- RGB-D 反投影与逐点 DINO/heat 对齐；
- FPS 确定性；
- 测地距离的统一尺度不变性；
- FGW 均匀边缘质量和 Contact Field 输运；
- 概率场不会被隐藏的 min-max 改写；
- 节点到完整可见部件的插值。

## 11. 当前限制与下一阶段接口

1. balanced FGW 会强制所有质量参与对应，严重遮挡、缺失或拓扑变化时可能产生
   错配；后续应把 unbalanced/partial FGW 作为显式消融，而不是静默替换。
2. 当前结构来自单视角可见 RGB-D，无法代表不可见面；完整表面传播必须留在后续
   抓取阶段。
3. FGW 非凸，固定 FPS seed 和 solver 参数保证工程复现，不代表全局最优。
4. 杯子需要补充源、目标 handle-only mask，才能完成严格功能部件对功能部件实验。
5. `alpha=0.2/0.5/0.8`、节点数、kNN 和 mask 质量需要在更多实例上系统消融。
6. Stage 1 的稳定输出接口是 `transfer_result.npz["target_heatmap"]`；后续模块只能
   消费保存结果，不能反向修改 AffCorrs/FGW 分数。

至此，LFV 第一阶段已经形成一个可独立运行、可配置替换、可拒绝失败、可保存全部
中间量并能被后续抓取复用的完整闭环。
