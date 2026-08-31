# LFV

LFV 当前包含两个通过保存文件解耦的阶段：

1. 无需训练的二维 one-shot affordance 迁移：用冻结 DINOv2 稠密描述符和
   Soft Heatmap AffCorrs，把源物体连续接触热力迁移到同类别目标 RGB 视角；
   可选 `affcorrs_fgw` 变体再利用源/目标完整功能部件 RGB-D 的 FGW 结构对应，
   恢复部件内部接触位置；
2. 可选的仿真抓取阶段：消费第一阶段的二维结果、ManiSkill 深度与完整 mesh，
   将热力提升到完整表面，再由 GraspNet 生成跨越热区两侧、近似 top-down 且
   通过严格点云碰撞筛选的抓取姿态。

另有一个通过保存文件消费前两阶段结果的动态验证层：加载历史训练完成的 pouring
GoalPose/Full64 模型，把杯子轨迹转换成 Panda TCP 轨迹，在 ManiSkill 中执行并保存
MP4、关键帧和分阶段诊断。首条 30 cm 杯碗间距基线见
`docs/deployment/aubo_camera_execution_bundle_zh.md`。

同一套配置驱动流程现已扩展到 drawer open：真实数据隔离、接触热力提取、
Soft Heatmap AffCorrs、完整把手抓取、Goal Pose/Full64 训练、64 坐标系叠加和
当前活动文档只维护 pouring 主流程和通用部署接口；drawer 等历史实验记录不再作为
当前实现的规范。

执行层支持 `panda_long_finger`：保留原 Panda 平行关节、80 mm 开口和 TCP，在两
个指 link 上加入 30×70 mm 高摩擦长指接触面。相同抓取和 Full64 轨迹已经从
“第 135 帧滑脱”改善为全程 `is_grasping=true`；当前剩余失败来自运动模型终点
没有到达远处 bowl。

drawer 使用同接口的 `panda_drawer_finger`：70 mm 长指板缩窄到 16 mm、厚度
4.5 mm，并相对 TCP 下移 30 mm；配合约 30 mm 预成形开口，可以从上方进入把手
前后间隙而不碰端部支撑或柜体顶板。权威 v2 配置和成功录像路径见 drawer 文档。

纯 Soft Heatmap AffCorrs 基线只回答“目标图像中的哪一块区域对应源任务接触
区域”，不会读取 depth/intrinsics。显式启用的 AffCorrs+FGW 变体会读取源、目标
可见功能部件 RGB-D 来恢复部件内位置，但仍不会读取仿真完整 mesh 或导入
GraspNet；完整表面和抓取仍是独立下游消费者。

## 固定真实验证

从仓库根目录运行：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/transfer_contact_heatmap.py \
  --config configs/affordance_transfer/episode0_to_maniskill.yaml
```

固定输入为：

- 源：`hand_pouring_lfv/episode_0` 第 39 帧、杯子掩码与连续接触热力；
- 目标：ManiSkill pouring snapshot 的 `rgb` 与 `cup_mask`；
- 特征：本地 `dinov2_vits14_pretrain.pth`，不会隐式联网下载。

固定输出目录：

```text
/home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/
└── episode_0_to_maniskill_seed_0/
    ├── transfer_result.npz
    ├── transfer_report.json
    └── transfer_summary.png
```

`transfer_summary.png` 同时显示源图与掩码、源连续热力、源正原型、目标图与
掩码、迁移后的目标连续热力，以及正向投票 V、反向验证 Q 和乘积 H。它是后续
快速迭代必须保持稳定的回归产物。

严格验证入口会在置信度拒绝时返回非零退出码：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/affordance_transfer/validate_episode0_to_maniskill.py
```

## 完整表面 Top-down 抓取

固定的端到端快速迭代入口为：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_maniskill_topdown.yaml
```

它依次运行二维迁移、RGB-D 反投影、完整表面反平行点对传播、GraspNet 解码与
top-down 细化、严格分部碰撞筛选和固定四联图生成。最终结果位于：

```text
/home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/
└── episode_0_to_maniskill_seed_0/topdown_grasp/
    ├── transferred_contact_3d.npz
    ├── graspnet_selected.npy
    ├── graspnet_selected_world.npy
    ├── graspnet_selected_object.npy
    ├── graspnet_full_contact_report.json
    ├── graspnet_selected_rgb_clean.png
    ├── graspnet_selected_open3d.png
    ├── topdown_grasp_summary.png
    └── topdown_grasp_pipeline_report.json
```

当前固定样本的选中抓取与世界 `-Z` 的夹角为 `6.37°`，左右接触热力为
`0.851/0.885`，接触对宽度为 `14.47 mm`；全局、左右手指、掌部和 approach
path 的点云碰撞 IoU 均为 `0`。这里验证的是静态点云碰撞，不等同于 IK、闭合
动力学或完整 pouring rollout 成功。

### 新杯子同类别泛化回归

仓库另提供一个外观和几何均不同的蓝色扫描杯固定案例，保持源帧、相机、杯碗
位置、随机种子和推理参数与基线一致：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_cole_blue_mug_topdown.yaml
```

新实例迁移置信度为 `0.4399`，热力正确位于左侧把手；最终抓取倾角为 `3.29°`，
接触对宽度 `23.89 mm`，所有分部碰撞 IoU 为 `0`。固定实例对比由
`scripts/sim/compare_topdown_grasp_instances.py` 生成，完整结论和局限见
`docs/stage1_affcorrs_fgw_contact_transfer_zh.md`。

## 测试

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python -m pytest -q
```

当前测试覆盖：

- 包围框留边、等比例缩放、letterbox 与坐标逆映射；
- 热力加权 K-Means 的确定性；
- Soft Heatmap AffCorrs 的正向/反向概率与乘积；
- RGB-D FGW 的 FPS、测地尺度归一化、均匀边缘质量和概率场输运；
- 循环一致性、峰值和熵置信度；
- 不读取目标深度/点云字段的二维端到端冒烟测试；
- RGB/heat/depth/intrinsics 的逐像素反投影；
- 可见热力到完整表面的反平行传播；
- 严格分部碰撞门限与固定四联抓取图生成。

## 当前代码结构

```text
configs/affordance_transfer/       算法默认配置与固定真实案例
configs/affordance_grasp/          固定完整表面、任务方向约束抓取配置
lfv/features/                      冻结稠密特征接口和 DINOv2 实现
lfv/affordance_transfer/           数据契约、预处理、聚类、匹配、置信度、I/O
lfv/lifting/                       二维 heat/depth 到相机点云的严格反投影
lfv/geometry/                      完整表面热力传播与反平行点对
lfv/grasping/                      抓取硬约束和严格碰撞筛选
lfv/visualization/                 固定二维迁移图与下游抓取四联图
scripts/affordance_transfer/       通用迁移入口和固定回归入口
scripts/sim/                       lifting、GraspNet、渲染与一键下游流水线
scripts/inference/                 任务无关 Goal Pose + Full64 仿真推理
scripts/robot/                     抓取闭合、轨迹执行与双视角录像
tests/                             单元测试与冒烟测试
docs/                              计算、接口、边界与改进路线
```

原 Joint Contact–Grasp Diffusion、单视角 HaMeR 抓取伪标签训练闭环及其训练期
Open3D 可视化已经退出活动代码；当前下游使用的是重新建立的仿真抓取可视化。
删除前的轻量源码备份位于：

```text
/home/users1/ljian/LFV_stage1_joint_diffusion_pre_soft_affcorrs_20260731.tar.gz
```

已有数据集和 `/home/users1/ljian/lfv_runs` 历史输出没有被删除。

## 文档

当前文档索引和推荐阅读顺序见 [`docs/README_zh.md`](docs/README_zh.md)。核心文档为：

- `docs/project_architecture_and_development_guide_zh.md`：代码结构与迭代约束；
- `docs/methods/LFV_complete_method_material_zh.md`：完整方法计算材料；
- `docs/stage1_affcorrs_fgw_contact_transfer_zh.md`：Stage 1 接触场迁移；
- `docs/stage2/current_method_complete_zh.md`：Stage 2 运动场和分层扩散；
- `docs/deployment/strict_camera_inference_zh.md`：RGB-D 严格推理入口；
- `docs/deployment/aubo_camera_execution_bundle_zh.md`：Aubo 实机交付和执行。

## 可复现性约束

- 源和目标采用同样的掩码包围框留边、等比例缩放和对称填充；
- patch 描述符逐向量 L2 归一化；
- 源正原型聚类使用连续热力作为样本权重；
- 目标聚类、K-Means 初始化和所有 NumPy/PyTorch 随机数由配置种子控制；
- 输出热力只在目标掩码内归一化到 `[0,1]`，同时保留未归一化的 raw score；
- 第一阶段报告明确记录没有使用目标深度、点云和抓取模块；
- 第二阶段只消费第一阶段的保存结果，所有坐标变换、传播和抓取阈值由独立配置
  固定，并把严格筛选前后的候选及分部碰撞诊断全部落盘。
