# AffCorrs + FGW Contact Field Transport 实现与双任务验证

更新日期：2026-08-03
状态：第一版已实现；抽屉与 Cole 蓝色杯真实配置、热力 A/B 图和 GraspNet
可视化均已跑通。

## 1. 修改目的与边界

原 Soft Heatmap AffCorrs 擅长回答“目标图中哪个语义部件对应源接触部件”，但其
目标 K-Means cluster 只携带区域语义，不保持部件内部的相对空间位置。因此源把手
中央的局部热力容易变成目标整个把手上的响应。本轮保留旧算法作为独立基线，新增
RGB-D FGW 模块恢复部件内部结构对应：

```text
source/target RGB + functional-part mask
                 │
                 ├── DINOv2 + Soft Heatmap AffCorrs (target K=64)
                 │          └── semantic region / optional soft gate
                 │
source/target aligned depth + intrinsics + the same full-part masks
                 │
                 ├── visible part point clouds + per-point DINO descriptors
                 ├── FPS 256 nodes + kNN geodesic distance matrices
                 ├── POT balanced FGW, alpha=0.5
                 └── transport source Contact Field with T
                                  │
                                  ▼
              paired target 2D Contact Field (not min-max rescaled)
                                  │
                                  ▼
     existing complete-surface propagation + top-down/collision GraspNet
```

本轮没有修改热力的完整表面反平行传播、GraspNet、碰撞筛选或 Open3D 渲染算法；
这些模块只消费新的 `transfer_result.npz["target_heatmap"]`。没有进行轨迹执行。

## 2. 关键参数不要混淆

- `matching.target_clusters: 64` 是用户指定的 AffCorrs 目标过聚类数量；
- `fgw.node_count: 256` 是 FGW 几何图节点数，不是 K-Means cluster 数；
- `fgw.alpha: 0.5` 表示 DINO 跨实例语义代价和部件内部结构代价等权起步；
- 两侧均使用均匀边缘质量，第一版是 balanced、non-entropic FGW；
- kNN 初始 `k=10`，只有图不连通时才递增，测地距离以 95% 分位距离归一化并
  截断到 `[0,1]`。

## 3. 数据、坐标和数值契约

新增 `RGBDPart`：

```text
depth_m       float32 [H,W]  RGB 对齐、单位 m
intrinsic_cv  float32 [3,3]  OpenCV pinhole K
part_mask     bool    [H,W]  整个可见功能部件，不是只有高热区
```

反投影坐标为 OpenCV camera frame：`+x` 向右、`+y` 向下、`+z` 向前：

\[
p(u,v)=\left[(u-c_x)z/f_x,\;(v-c_y)z/f_y,\;z\right]^T.
\]

源、目标原图像素先用已保存的 `CropTransform.original_to_input()` 映射到相同的
DINOv2 letterbox 输入，再在 patch 中心网格上双线性采样特征并逐点 L2 归一化。
FGW 必须读取完整 part mask 内的合法深度点；源热力只作为每个源点的标量属性，
不参与 FGW 点云裁剪。

语义代价：

\[
M_{ij}=\tfrac12(1-\langle f_i^s,f_j^t\rangle)\in[0,1].
\]

源、目标各自建立 kNN 图，以三维边长为权重计算全对最短路径，分别按自身距离
分位数归一化，所以结构项不依赖相机平移、旋转和绝对尺度。POT 求解：

\[
\min_T (1-\alpha)\langle M,T\rangle+
\alpha\sum_{ikjl}(D^s_{ik}-D^t_{jl})^2T_{ij}T_{kl},
\quad T\mathbf1=a,\;T^T\mathbf1=b.
\]

Contact Field 直接按软计划输运：

\[
h_j^t=\frac{\sum_iT_{ij}h_i^s}{\sum_iT_{ij}+\epsilon}.
\]

下采样节点热力以三邻点反距离插值恢复到目标 part 的全部有效深度像素。保存给
下游的概率不做每样本 min-max；`transfer_summary.png` 为了看清空间分布，单独
显示按峰值归一化的视图。杯子旧数据没有独立把手 mask，因此杯子配置在整杯 FGW
之后乘以 AffCorrs 软门控；抽屉已有准确的完整黑色把手 mask，不需要门控。

## 4. 模块结构

```text
lfv/affordance_transfer/
├── schema.py                  RGBDPart 和原二维数据契约
├── adapters.py                RealSense episode / simulator NPZ RGB-D adapter
├── pipeline.py                原 Soft Heatmap AffCorrs，保持独立
├── fgw_contact_transfer.py    lifting、FPS、geodesic、FGW、field transport
├── app.py                     method 配置分发
└── io.py                      schema v1/v2 保存和 scope 声明

lfv/visualization/
└── affordance_transfer.py     固定 2×4 AffCorrs/FGW A/B 图

configs/affordance_transfer/
├── drawer_episode60_handle_only_fgw_k64_to_maniskill_front.yaml
└── episode0_to_cole_blue_mug_fgw_k64.yaml

configs/affordance_grasp/
├── drawer_episode60_handle_only_fgw_k64_to_maniskill_front_topdown.yaml
└── episode0_to_cole_blue_mug_fgw_k64_topdown.yaml

tests/
└── test_fgw_contact_transfer.py
```

`method` 省略或设为 `soft_heatmap_affcorrs` 时行为与旧配置一致；只有显式设为
`affcorrs_fgw` 才读取 depth/intrinsics 和点云。若 POT 不可导入，等节点数情况下
存在一个确定性的 SciPy Frank-Wolfe 兜底；正式双任务结果使用 POT 0.9.7.post1。

## 5. 固定输出

每个 FGW 迁移目录保存：

```text
transfer_result.npz
  target_heatmap                 下游实际概率场
  target_heatmap_raw             未门控的 FGW 场
  affcorrs_target_heatmap        K=64 旧基线
  target_heatmap_fgw_raw
  transport                      [Ns,Nt]
  semantic_cost                  [Ns,Nt]
  source_geodesic/target_geodesic
  source/target node points and pixels

transfer_report.json             配置、solver、图统计、置信度和 scope
transfer_summary.png             固定 2×4 A/B 诊断图
transfer_source_target_2x2.png   下游实际 heat 的简洁四联图
```

抓取目录继续保存：

```text
transferred_contact_3d.npz
graspnet_full_contact_report.json
graspnet_selected_{camera,world,object}.npy
graspnet_selected_rgb_clean.png
graspnet_selected_open3d.png
topdown_grasp_summary.png
topdown_grasp_pipeline_report.json
```

## 6. 复现命令

抽屉迁移与抓取：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/drawer_episode60_handle_only_fgw_k64_to_maniskill_front.yaml

/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/drawer_episode60_handle_only_fgw_k64_to_maniskill_front_topdown.yaml \
  --skip-snapshot --skip-transfer
```

蓝杯迁移与抓取：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/episode0_to_cole_blue_mug_fgw_k64.yaml

/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_cole_blue_mug_fgw_k64_topdown.yaml \
  --skip-snapshot --skip-transfer
```

## 7. 2026-08-03 双任务结果

### 抽屉

```text
source/target valid part points      840 / 1278
FGW nodes                            256 / 256
POT FGW objective                    0.121223
AffCorrs > 0.5*peak support          791 pixels
FGW      > 0.5*peak support          620 pixels
selected top-down angle              0.3877 deg
contact pair width                   16.0007 mm
left/right tip heat                  0.8826 / 0.8826
strict max collision IoU             0.0
ranked feasible grasps               186
```

FGW 把 K=64 的分散响应收回到把手中部的连续区域；最终抓取跨越把手前后表面，
接近严格竖直向下且无点云碰撞。

### Cole 蓝色杯

```text
source/target valid object points    2937 / 10175
FGW nodes                            256 / 256
POT FGW objective                    0.123663
AffCorrs > 0.5*peak support          228 pixels
final FGW > 0.5*peak support         23 pixels
selected top-down angle              6.4693 deg
contact pair width                   22.7747 mm
left/right tip heat                  0.6295 / 0.4863
strict max collision IoU             0.0
ranked feasible grasps               1
```

蓝杯最终热力集中在左侧把手并产生一个通过严格碰撞检查的跨两侧抓取，但只剩一个
可行候选，鲁棒余量低于抽屉。主要原因不是 FGW 节点数，而是旧 pouring episode
只有整杯 mask：语义门控和整杯几何共同承担了把手定位。后续应优先离线补存源、
目标完整把手 mask，再做 `alpha=0.2/0.5/0.8` 与 node count 消融；不应通过降低
碰撞门限来掩盖功能部件 mask 的不足。

## 8. 测试

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python -m pytest -q
```

当前仓库 `37 passed`。FGW 专项测试覆盖：RGB-D/像素/特征/热力对齐、FPS
确定性、几何尺度不变性、均匀边缘质量、Contact Field 不被 min-max、节点热力
插值；原 AffCorrs、lifting、完整表面传播、抓取约束和报告测试也全部回归通过。

## 9. 当前限制

1. balanced FGW 假设源、目标可见部件质量均匀，严重遮挡或拓扑差异时可能强制
   错配；需要另开 unbalanced/partial OT 实验，不能静默替换当前基线。
2. 只使用单视角可见点云建立结构；本轮没有把仿真完整 mesh 偷渡进跨实例迁移，
   完整 mesh 仍只属于后续抓取阶段。
3. FGW 非凸，当前固定 seed/FPS/POT 参数保证工程复现，但不代表全局最优。
4. 杯子缺 handle-only mask，当前结果可用于闭环验证，但不能等同于“严格的完整
   功能部件到完整功能部件 FGW”实验。
5. 本轮验证的是静态抓取位姿和点云碰撞，不是闭合动力学、IK 或任务执行成功率。
