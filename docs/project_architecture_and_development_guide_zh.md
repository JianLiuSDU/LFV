# LFV 当前架构与快速迭代开发指南

更新日期：2026-08-03

## 1. 当前三个通过保存文件解耦的阶段

LFV 第一阶段默认实现二维连续 affordance 迁移，并提供一个显式启用的 RGB-D
结构对应变体：

```text
source RGB + source mask + source continuous heat
                        │
                        ▼
       frozen DINOv2 + Soft Heatmap AffCorrs
                        │
                        ▼
target RGB + target mask + target continuous heat + confidence
```

`method: affcorrs_fgw` 时，AffCorrs 只负责语义区域，完整源/目标功能部件 RGB-D
经 FPS、kNN 测地距离和 balanced FGW 迁移部件内部 Contact Field。旧
`soft_heatmap_affcorrs` 仍然完全不读深度；两种方法由同一入口和配置分发，详见
`docs/stage1_affcorrs_fgw_contact_transfer_zh.md`。

目标是确认“人类演示中接触的语义部件能否迁移到同类别仿真实例的相机
图像”。这一阶段不训练 Joint Contact–Grasp Diffusion，也不从单视角手部关键点
构造夹爪标签。纯二维基线明确排除：

- 目标深度和相机内参；
- 二维到三维反投影；
- 单侧可见热力向不可见表面的扩散；
- 完整点云、Open3D 和 GraspNet。

第二阶段是第一阶段的可选下游消费者：

```text
saved target heat + target depth/intrinsics + simulator complete mesh
                                │
                                ▼
     pixel-aligned lifting + antipodal complete-surface propagation
                                │
                                ▼
  GraspNet proposals + top-down refinement + strict collision filtering
                                │
                                ▼
        camera/world/object grasp pose + fixed visual report
```

第二阶段会读取深度、相机内参、完整杯面和场景点云，但不会修改二维迁移分数，
也不会让三维信息进入 `soft_affcorrs.py`。

第三阶段是 2026-08-01 新增的动态验证消费者：读取保存的 GraspNet object-frame
抓取、历史训练完成的 pouring GoalPose/Full64 权重及新场景 snapshot，将预测的杯子
SE(3) 轨迹转换为 Panda TCP 轨迹，通过 ManiSkill `pd_ee_pose` 执行并固定保存录像、
关键帧和诊断指标。它不会反向修改二维热力或静态 GraspNet 结果。DIFT、多源帧融合、
CRF 和学习式点云补全仍未加入。

## 2. 活动目录

```text
LFV/
├── configs/
│   ├── affordance_transfer/
│   │   ├── soft_heatmap_affcorrs.yaml
│   │   └── episode0_to_maniskill.yaml
│   └── affordance_grasp/
│       ├── episode0_to_maniskill_topdown.yaml
│       └── episode0_to_cole_blue_mug_topdown.yaml
├── lfv/
│   ├── features/
│   │   ├── base.py
│   │   └── dinov2_dense.py
│   ├── affordance_transfer/
│   │   ├── schema.py
│   │   ├── preprocessing.py
│   │   ├── clustering.py
│   │   ├── soft_affcorrs.py
│   │   ├── confidence.py
│   │   ├── adapters.py
│   │   ├── pipeline.py
│   │   ├── fgw_contact_transfer.py
│   │   ├── io.py
│   │   └── app.py
│   ├── lifting/
│   │   └── image_heat_to_surface.py
│   ├── geometry/
│   │   └── contact_heat_propagation.py
│   ├── grasping/
│   │   └── constraints.py
│   └── visualization/
│       ├── affordance_transfer.py
│       └── topdown_grasp_report.py
│   ├── inference/functional_motion/
│   │   └── two_stage_pouring.py
│   └── robot/
│       └── panda_grasp_execution.py
├── scripts/
│   ├── affordance_transfer/
│   │   ├── transfer_contact_heatmap.py
│   │   └── validate_episode0_to_maniskill.py
│   └── sim/
│       ├── export_pouring_contact_snapshot.py
│       ├── lift_transferred_heat_to_complete_surface.py
│       ├── generate_graspnet_from_full_contact.py
│       ├── render_pouring_contact_camera_view.py
│       ├── compose_topdown_grasp_summary.py
│       ├── compare_topdown_grasp_instances.py
│       └── run_transferred_heat_topdown_grasp.py
│   ├── inference/infer_pouring_motion.py
│   ├── robot/execute_pouring_motion_maniskill.py
│   └── run_pouring_motion_execution.py
├── tests/
│   ├── test_affordance_transfer_preprocessing.py
│   ├── test_weighted_kmeans.py
│   ├── test_soft_affcorrs.py
│   ├── test_affordance_transfer_confidence.py
│   ├── test_fgw_contact_transfer.py
│   ├── test_episode0_maniskill_transfer_smoke.py
│   ├── test_image_heat_lifting.py
│   ├── test_contact_heat_propagation.py
│   ├── test_topdown_grasp_constraints.py
│   └── test_topdown_grasp_report.py
└── docs/
    ├── soft_heatmap_affcorrs_refactor_plan_zh.md
    ├── transferred_heat_topdown_grasp_zh.md
    ├── cole_blue_mug_generalization_validation_zh.md
    └── project_architecture_and_development_guide_zh.md
```

旧的 RGB-D 数据处理模块仍可用于生成源 RGB、物体掩码和连续接触热力，但它们
不是当前迁移算法本身。仿真点云和 GraspNet 只属于独立的第二阶段活动链路。

## 3. 数据契约

### 3.1 源样本

`SourceContactExample`：

```text
rgb       uint8   [Hs, Ws, 3]  RGB 顺序
mask      bool    [Hs, Ws]     完整源物体前景
heatmap   float32 [Hs, Ws]     [0,1] 连续任务接触热力
sample_id str
```

加载后执行：

```text
heatmap = clip(heatmap, 0, 1) * mask
```

源热力不能为空。当前固定样本为：

```text
/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0
frame_index = 39
mask = sam_mask/affordance_mask.npy
heat = contact_heatmap/contact_heatmap.npz["heatmap_2d"]
```

### 3.2 目标样本

`TargetObservation`：

```text
rgb       uint8 [Ht, Wt, 3]
mask      bool  [Ht, Wt]
sample_id str
```

固定目标为：

```text
/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/
  seed_0_dataset_aligned/pouring_snapshot.npz
```

二维 adapter 只读取 `rgb` 与 `cup_mask`。即使 NPZ 同时含 `depth_m`、
`intrinsic_cv`、完整点云和抓取结果，第一阶段也不会访问这些字段；只有显式运行
第二阶段脚本后，下游 lifting 才重新打开该 snapshot 并读取三维字段。

FGW 配置使用独立 `RGBDPart` adapter 显式读取对齐的 `depth_m`、`intrinsic_cv`
和完整功能部件 mask。其几何只用于跨实例结构对应；不可见面完整 mesh 依然只
允许后续抓取阶段消费。

### 3.3 输出

`TransferResult`：

```text
target_heatmap      float32 [Ht, Wt]  掩码内归一化到 [0,1]
target_heatmap_raw  float32 [Ht, Wt]  未做样本内 min-max 的匹配分数
confidence          dict              global/cycle/peak/entropy 等
accepted            bool
rejection_reasons   list[str]
diagnostics         dict              网格图、数量、坐标变换和元数据
```

`diagnostics` 同时记录 source/target heat 的 `peak_uv`、加权
`centroid_uv` 和 heat mass，便于批量回归时不打开图也能检查峰值漂移。

## 4. 完整计算流程

### 4.1 相同的空间预处理

源和目标必须使用同一规则：

1. 计算物体 mask 的紧包围框；
2. 按宽、高各增加 `bbox_margin=0.15`；
3. 保持长宽比缩放；
4. 对称填充到 `518×518`；
5. 保存原图、crop、resize 和 padding 的双向坐标变换。

RGB 用双线性插值，mask 用最近邻，连续热力用双线性。mask 与 heat 不允许走
不同 crop，也不能在进入特征网格后再猜测坐标。

### 4.2 DINOv2 稠密描述符

默认使用本地冻结权重：

```text
model = vit_small_patch14_dinov2
patch = 14
input = 518 × 518
feature grid = 37 × 37
```

`DinoV2DenseExtractor.extract(rgb)` 返回：

```text
features float32 [37, 37, D]
```

每个 patch 描述符沿 `D` 做 L2 归一化。模型不会自动联网下载，也不会参与
训练。以后替换 DIFT 或其他特征时，只需实现 `DenseFeatureExtractor`：

```python
class DenseFeatureExtractor(Protocol):
    patch_size: int
    def extract(self, rgb: np.ndarray) -> np.ndarray: ...
```

### 4.3 源热力正原型

先把源 mask 与 heat 以同一空间映射缩放到 DINO 网格。对源前景 patch：

```text
P = {i | A_s(i) > tau_pos}
```

只在 `P` 内运行热力加权 K-Means：

```text
min Σ_i A_s(i) ||f_i - z^s_{c_i}||²
```

源原型权重为：

```text
omega_k =
  Σ_{i in C_k} A_s(i)
  -------------------
  Σ_{i in P} A_s(i)
```

同时记录阈值以上保留的总热力比例，避免阈值过高却仍然输出看似自信的结果。

### 4.4 目标前景过分割

在目标 mask 内所有 DINO patch 上运行更密集的普通 K-Means，默认
`K_target=64`。得到目标区域原型 `z^t_j`，每个前景 patch 保留其 cluster
label。

### 4.5 正向区域投票

源接触原型与目标区域原型计算余弦相似度：

```text
S_kj = <z^s_k, z^t_j>
P^f_kj = softmax_j(S_kj / T_f)
V_j = Σ_k omega_k P^f_kj
```

`V_j` 表示源接触语义对目标区域的正向支持。

### 4.6 反向整物体验证

每个目标原型反向匹配到**源物体全部前景 patch**，不是只匹配源热区：

```text
P^b_ji = softmax_i(<z^t_j, f^s_i> / T_b)
Abar_s(i) = A_s(i) / Σ_i A_s(i)
Q_j = Σ_i Abar_s(i) P^b_ji
```

如果一个目标区域正向看起来相似，但反向主要落在源杯身等非任务区域，`Q_j`
会压低它。

### 4.7 最终连续热力

```text
H_j = V_j * Q_j
```

把 `H_j` 分配给目标 cluster 内所有 patch，双线性插值回预处理输入，再通过
保存的逆坐标映射还原到原始目标图。最后：

```text
target_heatmap_raw = mapped_H * target_mask
target_heatmap = minmax(target_heatmap_raw inside target_mask)
```

不进行二值化，不做 CRF。

## 5. 置信度与拒绝

报告至少包含：

- `cycle`：反向热力落回概率相对 uniform baseline 的校准值；
- `peak`：目标正分数 95% 与 50% 分位数的显著度；
- `entropy`：目标 patch 热力分布的归一化熵；
- `retained_heat_mass`：源正阈值保留的热力比例；
- `global`：cycle、peak 和 `1-entropy` 的几何平均；
- `retained_heat_mass` 单独报告和拒绝，不参与 global，避免它抬高匹配置信度。

任何一项越过配置的拒绝阈值，都会记录明确原因。通用迁移入口仍保存结果，
便于诊断；严格验证入口在拒绝时返回退出码 2，适合快速回归和自动化。

## 6. 保存与可视化

每次运行固定生成：

```text
transfer_result.npz
transfer_report.json
transfer_summary.png
```

NPZ 保存目标 heat raw/normalized，以及源原型、目标 cluster、V/Q/H 网格。
JSON 保存参数、坐标变换、置信度和作用域声明，并明确：

```json
{
  "uses_target_depth": false,
  "uses_point_cloud": false,
  "uses_graspnet": false
}
```

PNG 是固定 2×3 回归图：

1. 源 RGB + object mask；
2. 源连续热力；
3. 源热力正 patch 与原型；
4. 目标 RGB + object mask；
5. 迁移的目标连续热力；
6. V、Q、H 三张网格图。

可视化不负责重新计算算法，也不读取配置或深度。

## 7. 快速迭代命令

通用入口：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/episode0_to_maniskill.yaml
```

覆盖输出目录或设备：

```bash
.../python scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/episode0_to_maniskill.yaml \
  --output-dir /tmp/affcorrs_trial \
  --device cuda
```

严格固定回归：

```bash
.../python scripts/affordance_transfer/validate_episode0_to_maniskill.py
```

## 8. 测试职责

```text
test_affordance_transfer_preprocessing.py
  letterbox、mask/heat 对齐、坐标 round-trip、逆映射

test_weighted_kmeans.py
  加权聚类、确定性、cluster mass

test_soft_affcorrs.py
  正向/反向 softmax、热力正集、语义区域优先级

test_affordance_transfer_confidence.py
  cycle/peak/entropy/global 与拒绝原因

test_episode0_maniskill_transfer_smoke.py
  模块组合、输出 shape/mask、目标 adapter 不读深度/点云

test_fgw_contact_transfer.py
  RGB-D 对齐、FPS、测地尺度不变、FGW 边缘质量与概率场输运
```

新增算法必须先增加最小合成测试，再运行真实固定案例并检查 PNG。只报告命令
成功而不检查热力是否位于目标语义部件，不算完成。

第二阶段的附加测试职责：

```text
test_image_heat_lifting.py
  RGB/heat/depth/intrinsics 的像素对齐反投影、热阈值不改变可见几何

test_contact_heat_propagation.py
  完整表面传播、反平行法向和隐藏侧点对

test_topdown_grasp_constraints.py
  official collision flag 与 global/finger/palm/path 五类硬门限

test_topdown_grasp_report.py
  固定四联图的无 GUI 合成与落盘

test_pouring_motion_execution_geometry.py
  256/64 点输入契约、局部/相机/世界坐标、GraspNet/Panda 轴约定和刚性附着
```

## 9. 删除与备份记录

以下旧活动闭环已在 2026-07-31 删除：

- Joint Contact–Grasp Diffusion 的 dataset、PointNet++、diffusion、model
  registry、trainer、evaluation、checkpoint、采样和 Open3D 可视化；
- 相应 YAML、单元测试、overfit 测试和文档；
- 从单视角 HaMeR thumb-index 关键点构造夹爪伪标签的活动 stage；
- 依赖旧预测模型的 ManiSkill contact-to-GraspNet 集成入口。

没有删除 `/media/ljian/lj` 数据和已有 `lfv_runs` 结果。删除前源码备份：

```text
/home/users1/ljian/LFV_stage1_joint_diffusion_pre_soft_affcorrs_20260731.tar.gz
```

## 10. 第二阶段固定运行流程

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_maniskill_topdown.yaml
```

它固定执行：二维迁移 → 逐像素反投影 → 完整表面反平行点对传播 → GraspNet
解码 → top-down 点对细化 → 标准和严格分部碰撞筛选 → Open3D/Xvfb 与四联图。
完整公式、坐标系、阈值、当前真实结果和输出契约见
`docs/stage1_affcorrs_fgw_contact_transfer_zh.md`。

`snapshot_export` 是可选配置层：YCB 资产可按 model id 加载，自定义扫描杯使用
visual mesh 加若干 convex collision mesh。当前受控新实例回归为
`Cole_Hardware_Mug_Classic_Blue`，结果与对比入口见
运行报告保存在对应的 `lfv_runs` 输出目录中。

## 11. 后续接入规则

二维核心接口 `pipeline.transfer(source, target)` 与输出 schema 必须继续保持不变。
后续优先级为：

1. 验证更多 source-target 同类别实例和不同视角；
2. 加入同一演示多源帧、按置信度融合；
3. 比较 DINOv2、DIFT 或两者分数融合；
4. 用 `panda_long_finger` 与状态探针回归夹持稳定性，保持 TCP/关节接口不变；
5. 对 GoalPose 多 seed 采样做目标距离、可达性和碰撞筛选；
6. 统计多场景完整 rollout 成功率，再决定运动模型重训或域适配。

任何扩展都应通过保存文件消费上游结果，不能把 depth、点云传播或 GraspNet
逻辑塞回 `soft_affcorrs.py`。

## 12. 动态推理、执行与录像基线

固定入口为：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/run_pouring_motion_execution.py \
  --config configs/experiments/functional_motion/pouring_cup_far_execution.yaml
```

模型选择、256/64 点真实 shape、坐标变换、30 cm 场景、视频产物、首次失败结果
及执行边界记录在部署文档中。执行器已
明确固定 Panda 组合动作的混合单位：arm 是未归一化绝对位姿，gripper 是归一化
标量 `+1=全开/-1=全闭`；同时逐帧同步保存斜前方和正前方 `base_camera` 两条录像。
机器人资产层新增可选 `panda_long_finger`：由
`lfv/robot/gripper_extension.py` 固定独立几何契约，由
`lfv_sim/maniskill/robots/panda_long_finger.py` 向原 Panda 两个 finger link 同时添加
视觉、碰撞和高摩擦接触面。当前 30×70 mm 接触面是原指垫的 6.49 倍，完整 64 步
轨迹已保持抓取到结束；详情和真实硬件边界见执行基线文档 4.3 节。

drawer v2 复用同一执行接口，但使用 `panda_drawer_finger`（16×70×4.5 mm、沿
手指轴额外下移 30 mm）与 `approach_gripper_action=0.0` 的约 30 mm 预成形开口，
从上方进入把手前后间隙后才发送 `-1.0` 完全闭合。场景固定为 table-front
`base_camera`、drawer yaw=0、世界 +X 开启轴；配置、接触对硬约束与成功录像见
具体历史录像不属于当前活动文档。
