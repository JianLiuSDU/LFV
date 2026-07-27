# LFV 项目结构与开发说明

## 1. 文档目的

本文档说明 LFV 重构后的项目结构、各目录职责、模块之间的依赖关系，
以及后续如何在同一套框架下完成：

1. 人类 RGB-D 操作视频的数据处理与标签生成；
2. Contact-Grasp Generation Network 的训练与推理；
3. Functional Motion Generation Network 的训练与推理；
4. 两阶段模型的组合推理；
5. 机器人运动学筛选；
6. 仿真场景中的闭环执行、可视化和评估。

当前仓库已经保留了可用的数据处理代码，但新的模型、训练框架和仿真接口
尚未正式实现。本文档中的“建议新增”目录应在对应功能开始开发时再创建，
不需要提前放置大量空文件。

## 2. 总体设计目标

LFV 的目标不是针对 pouring 单独构造一套代码，而是建立一个能够支持多种
人类操作任务的统一框架，例如：

- 杯子向碗中倒水；
- 打开抽屉；
- 将物体放入容器；
- 按压按钮；
- 搅拌、扫动和推拉操作。

完整系统分为四个层次：

```text
人类 RGB-D 视频
    |
    v
离线数据处理与伪标签生成
    |
    +--> 物体点云、DINO 特征、接触热力、抓取伪标签
    |
    +--> 被操作物体相对于目标物体的 SE(3) 功能轨迹
    |
    v
Stage 1: Contact-Grasp Generation
    |
    +--> 任务相关逐点接触热力
    +--> 与接触区域匹配的抓取候选
    |
    v
Stage 2: Functional Motion Generation
    |
    +--> 多模态物体 SE(3) 功能运动轨迹
    |
    v
机器人可执行性筛选
    |
    +--> 坐标转换、连续 IK、碰撞、关节限位和平滑性评分
    |
    v
仿真或真实机器人执行
```

两个生成模型应保持解耦：

- Stage 1 学习“抓哪里”和“如何抓”；
- Stage 2 学习“物体应该如何运动”；
- `lfv/robot/` 在后验阶段组合抓取与轨迹，并判断机器人是否能够执行。

这样可以独立替换接触模型、抓取模型、轨迹模型、机器人型号或仿真后端。

## 3. 数据与代码的存放原则

代码仓库只保存：

- Python 源代码；
- YAML 配置；
- 测试；
- 文档；
- 小型示例元数据；
- 第三方源码或指向第三方仓库的链接。

所有大规模数据、模型权重和运行结果放在 `/media/ljian/lj` 下。推荐约定：

```text
/media/ljian/lj/
├── hand_data/                  # 原始人类 RGB-D 视频
├── data_3d/                    # 处理后的训练数据和标签
├── lfv_checkpoints/            # 模型 checkpoint
├── lfv_model_outputs/          # 离线推理结果
├── lfv_eval_outputs/           # 数据集评估结果
└── lfv_sim_outputs/            # 仿真 rollout、视频和评估报告
```

当前 pouring 数据路径为：

```text
原始数据：
/media/ljian/lj/hand_data/pouring

处理结果：
/media/ljian/lj/data_3d/hand_pouring_lfv
```

任何 Dataset、训练脚本或仿真程序都不能默认向 LFV 代码目录写入数据。

## 4. 推荐项目结构

当前结构基础上，模型和仿真实现逐步补齐后，推荐形成以下目录：

```text
LFV/
├── configs/
│   ├── data/
│   ├── pipeline/
│   ├── model/                         # 建议新增
│   ├── training/                      # 建议新增
│   ├── inference/                     # 建议新增
│   ├── simulation/                    # 建议新增
│   └── experiments/
│
├── docs/
├── legacy/
│
├── lfv/
│   ├── data_processing/
│   ├── pipeline/
│   ├── datasets/
│   ├── models/
│   │   ├── common/
│   │   ├── contact_grasp/
│   │   └── functional_motion/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── robot/
│   ├── simulation/                    # 开始仿真时新增
│   ├── visualization/
│   └── utils/
│
├── scripts/
│   ├── preprocess/
│   ├── train/
│   ├── infer/
│   ├── evaluate/
│   ├── visualize/
│   ├── robot/
│   └── simulate/                      # 开始仿真时新增
│
├── tests/
├── third_party/
├── tools/
├── README.md
├── pyproject.toml
└── requirements.txt
```

目录名称建议统一使用：

```text
contact_grasp
functional_motion
```

后续可将当前较长的 `contact_grasp_generation` 和
`functional_motion_generation` 简化为上述名称，避免模型、训练和配置中
出现多种不同命名。

## 5. 顶层目录职责

### 5.1 `configs/`

`configs/` 是所有可复现实验参数的唯一来源。代码中不应散落数据路径、
网络维度、噪声步数、阈值和 checkpoint 路径。

推荐配置分组：

```text
configs/
├── data/
│   ├── datasets/               # 数据位置、artifact 名称和质量过滤规则
│   └── tasks/                  # 任务语义、对象角色和坐标系约定
├── pipeline/                   # RGB-D 离线处理参数
├── model/
│   ├── contact_grasp/
│   └── functional_motion/
├── training/                   # optimizer、batch、epoch、EMA、日志
├── inference/                  # 采样数量、guidance、候选过滤
├── simulation/                 # 仿真后端、机器人、场景和 rollout 参数
└── experiments/                # 对以上配置的组合和少量实验覆盖
```

模型组件建议使用 Hydra `_target_` 构造：

```yaml
model:
  _target_: lfv.models.contact_grasp.model.ContactGraspModel

  point_encoder:
    _target_: lfv.models.common.pointcloud.point_encoder.PointEncoder
    hidden_dim: 256

  contact_generator:
    _target_: lfv.models.contact_grasp.contact_diffusion.ContactDiffusion
    prediction_type: epsilon
```

更换编码器或 denoiser 时只修改配置，不修改训练脚本。

### 5.2 `docs/`

保存稳定的项目说明和设计决策：

- 数据格式；
- 坐标系约定；
- 标签生成方法；
- 模型设计；
- 实验协议；
- 仿真接口；
- 当前完成状态和交接说明。

一次性调试记录不应长期堆积在 `docs/` 中。

### 5.3 `legacy/`

只保存旧版本的位置说明和迁移说明，不再放置旧模型副本。

旧代码备份当前位于：

```text
/home/users1/ljian/LFV_legacy_20260727_no_data_no_third_party
```

### 5.4 `third_party/`

保存或链接外部依赖，例如：

- SAM2；
- TAPIP3D / CoTracker；
- HaMeR；
- DINOv2 权重；
- 后续可能使用的 GraspNet、AnyGrasp 或仿真依赖。

LFV 自己的代码通过适配器调用第三方实现，不直接修改第三方源码。
具有冲突依赖的第三方项目应使用独立 Conda 环境运行，并通过文件、中间
结果或子进程与 LFV 主环境通信。

### 5.5 `tools/`

用于一次性检查、数据诊断和实验性工具，例如：

- 检查一个 episode 的字段；
- 查看接触热力；
- 验证 HaMeR 伪抓取；
- 分析 TCP 深度；
- 手工查看 Open3D 可视化。

当一个工具成为稳定工作流后：

1. 核心实现移动到 `lfv/`；
2. 正式入口移动到 `scripts/`；
3. `tools/` 中只保留诊断版本或删除重复工具。

### 5.6 `scripts/`

只放用户直接执行的薄入口。脚本负责：

- 解析命令行和 Hydra 配置；
- 创建运行目录；
- 调用 `lfv/` 中的实现；
- 返回退出码和简短摘要。

脚本中不应实现网络、损失、数据格式转换或主要算法。

建议最终只保留少量通用入口：

```text
scripts/preprocess/run_pipeline.py
scripts/train/train.py
scripts/infer/infer.py
scripts/evaluate/evaluate.py
scripts/simulate/run_rollout.py
scripts/visualize/visualize_episode.py
```

不同模型和任务通过配置切换，避免为每个实验复制一个 Python 脚本。

## 6. `lfv/` 包内部职责

### 6.1 `lfv/pipeline/`

这是当前已经验证的数据处理实现，包含：

- RGB-D episode 准备；
- GroundingDINO 检测；
- SAM2 物体分割；
- 点云采样；
- 点跟踪；
- SE(3) 轨迹恢复；
- 手部检测和分割；
- 接触时间判断；
- 接触热力生成；
- DINOv2 特征提取；
- HaMeR 手部姿态；
- 拇指—食指抓取伪标签。

该目录属于离线数据生产，不允许被模型 `forward()` 调用。

### 6.2 `lfv/data_processing/`

这个目录提供稳定的数据处理 API 和 episode 读写接口。

短期内：

- `lfv/pipeline/` 保留已验证算法；
- `lfv/data_processing/` 提供统一入口、数据 schema 和 I/O。

长期内应逐步消除二者职责重叠。建议最终将处理器组织为：

```text
lfv/data_processing/
├── episode_io.py
├── schemas.py
├── rgbd.py
├── geometry.py
├── object_segmentation.py
├── hand_processing.py
├── contact_labeling.py
├── grasp_labeling.py
├── dino_features.py
└── pipeline_runner.py
```

迁移必须逐项验证，不能一次性移动当前可运行代码。

### 6.3 `lfv/datasets/`

Dataset 只读取已经完成处理的 artifact，并将它们整理成张量。
Dataset 不运行 SAM2、HaMeR、DINO 或点跟踪。

建议文件：

```text
lfv/datasets/
├── schemas.py
├── artifact_store.py
├── contact_grasp_dataset.py
├── functional_motion_dataset.py
├── transforms.py
├── normalization.py
└── collate.py
```

职责：

- `schemas.py`：定义 batch 中字段名称、形状、dtype 和坐标系；
- `artifact_store.py`：统一读取 `.npz`、`.npy`、JSON、Zarr；
- `contact_grasp_dataset.py`：提供 Stage 1 输入和监督；
- `functional_motion_dataset.py`：提供 Stage 2 输入和监督；
- `transforms.py`：点采样、数据增强和随机遮挡；
- `normalization.py`：记录并应用坐标、轨迹和 DINO 特征归一化；
- `collate.py`：处理变长轨迹、候选数量和有效掩码。

Stage 1 建议数据契约：

```text
points_object_m          [N, 3]
points_object_norm       [N, 3]
normals_object           [N, 3]
dinov2_point_features    [N, C]
contact_heat             [N, 1]
grasp_translation        [K, 3]
grasp_rotation_6d        [K, 6]
grasp_width              [K, 1]
grasp_valid_mask         [K]
```

Stage 2 建议数据契约：

```text
manipulated_points       [Nm, 3]
reference_points         [Nr, 3]
manipulated_features     [Nm, C]
reference_features       [Nr, C]
trajectory_translation   [T, 3]
trajectory_rotation_6d   [T, 6]
trajectory_valid_mask    [T]
```

所有字段必须明确单位和坐标系，不能只通过变量名猜测。

### 6.4 `lfv/models/common/`

只放至少被两个模型实际复用的组件。

建议结构：

```text
lfv/models/common/
├── diffusion/
│   ├── noise_schedule.py
│   ├── timestep_embedding.py
│   ├── sampler.py
│   └── guidance.py
├── geometry/
│   ├── rotation_6d.py
│   ├── so3.py
│   └── se3.py
├── pointcloud/
│   ├── point_encoder.py
│   ├── neighborhood.py
│   └── feature_fusion.py
└── attention/
    └── cross_attention.py
```

不要在第一次实现时为了“看起来通用”而提前抽象。一个模块在两个阶段中
真正出现相同需求后，再移动到 `common/`。

### 6.5 `lfv/models/contact_grasp/`

Stage 1 的所有任务相关模型代码放在同一目录：

```text
lfv/models/contact_grasp/
├── model.py
├── conditioning.py
├── contact_denoiser.py
├── contact_diffusion.py
├── grasp_denoiser.py
├── grasp_diffusion.py
├── losses.py
├── sampling.py
└── types.py
```

各文件职责：

- `model.py`
  - 组合点云编码器、接触生成器和抓取生成器；
  - 对外提供统一的 `compute_loss()` 和 `sample()`；
  - 不负责 optimizer、日志和 checkpoint。

- `conditioning.py`
  - 融合 XYZ、法向、DINO、尺度和任务条件；
  - 从接触热力中聚合高热区域特征；
  - 构造 classifier-free guidance 条件。

- `contact_denoiser.py`
  - 接收带噪逐点热力、扩散时间步和点云条件；
  - 输出噪声、`x0` 或 `v` 预测。

- `contact_diffusion.py`
  - 实现热力加噪、训练目标和采样；
  - 处理 `[0,1]` 标签映射、点有效掩码和后处理。

- `grasp_denoiser.py`
  - 预测抓取平移、rotation-6D 和夹爪宽度的噪声；
  - 使用点云和已生成接触区域作为条件。

- `grasp_diffusion.py`
  - 实现抓取变量的归一化、加噪、采样和合法范围恢复。

- `losses.py`
  - Stage 1 特有损失；
  - 接触监督、抓取监督、接触—抓取一致性和合法性损失。

- `sampling.py`
  - 多接触样本和多抓取候选的组合；
  - 候选去重、基础几何过滤和输出格式整理。

- `types.py`
  - 定义模型输出结构，例如 `ContactSample`、`GraspCandidate`。

建议将接触场和抓取生成器拆开，再由 `ContactGraspModel` 组合。这样可以：

- 先训练接触场；
- 使用真实接触标签训练抓取模型；
- 使用生成接触场训练抓取模型；
- 将抓取 diffusion 替换为回归 head 或其他生成模型；
- 最后进行联合微调。

### 6.6 `lfv/models/functional_motion/`

Stage 2 推荐结构：

```text
lfv/models/functional_motion/
├── model.py
├── object_encoder.py
├── relation_encoder.py
├── trajectory_denoiser.py
├── trajectory_diffusion.py
├── losses.py
├── sampling.py
└── types.py
```

职责：

- `object_encoder.py`
  - 编码被操作物体和参考物体点云；

- `relation_encoder.py`
  - 表达两个物体之间的相对位置、语义和几何关系；

- `trajectory_denoiser.py`
  - 对 `[T, 3+6]` 的平移和 rotation-6D 轨迹去噪；

- `trajectory_diffusion.py`
  - 实现轨迹加噪、mask、采样和反归一化；

- `losses.py`
  - 轨迹重建、旋转、速度、加速度和关系一致性损失；

- `sampling.py`
  - 多模态轨迹采样、去重和基本物理过滤；

- `model.py`
  - 对外提供 `compute_loss()` 和 `sample()`。

轨迹始终在统一的初始被操作物体坐标系下表示，不在模型中直接绑定机器人
基座坐标系。

### 6.7 `lfv/training/`

只保存与具体任务无关的训练基础设施：

```text
lfv/training/
├── trainer.py
├── checkpoint.py
├── ema.py
├── optimizer.py
├── distributed.py
├── logging.py
└── seed.py
```

通用 Trainer 调用模型的：

```python
losses = model.compute_loss(batch)
```

Trainer 不应该判断当前模型是 Contact-Grasp 还是 Functional Motion。
这能避免为两个阶段复制两套训练循环。

checkpoint 至少保存：

- 模型权重；
- EMA 权重；
- optimizer 和 scheduler；
- 当前 epoch / step；
- 完整解析后的配置；
- 数据集版本或清单哈希；
- 归一化统计量；
- Git commit。

### 6.8 `lfv/inference/`

负责加载 checkpoint 和编排推理，不重复实现 diffusion 采样算法。

推荐文件：

```text
lfv/inference/
├── checkpoint_loader.py
├── contact_grasp.py
├── functional_motion.py
└── two_stage_pipeline.py
```

- `contact_grasp.py` 调用 Stage 1 的 `model.sample()`；
- `functional_motion.py` 调用 Stage 2 的 `model.sample()`；
- `two_stage_pipeline.py` 组合两阶段输出并交给 `lfv/robot/`。

标准两阶段推理过程：

```text
输入场景点云
  -> 生成 S 个 Contact Field
  -> 每个 Contact Field 生成 G 个抓取
  -> 生成 M 条功能运动轨迹
  -> 形成 S x G x M 个组合
  -> 机器人运动学与碰撞筛选
  -> 输出 Top-K 可执行方案
```

候选数量必须通过配置控制，并使用逐级筛选避免组合爆炸。

### 6.9 `lfv/evaluation/`

只负责指标，不负责训练和模型结构。

推荐文件：

```text
lfv/evaluation/
├── contact_metrics.py
├── grasp_metrics.py
├── motion_metrics.py
├── feasibility_metrics.py
└── aggregation.py
```

建议指标：

- Contact：
  - point-wise MSE / BCE；
  - Top-k contact coverage；
  - 高热区域 IoU；
  - 接触区域中心和表面距离；

- Grasp：
  - 抓取接触点与热区一致性；
  - 姿态和宽度误差；
  - 碰撞率；
  - grasp success 或仿真成功率；
  - 多样性和覆盖率；

- Motion：
  - 平移和旋转误差；
  - DTW 或时间对齐误差；
  - 多模态 min-of-N 指标；
  - 目标物体关系完成度；

- Robot：
  - 连续 IK 成功率；
  - 碰撞率；
  - 关节限位余量；
  - 轨迹平滑性；
  - 完整任务成功率。

### 6.10 `lfv/robot/`

只处理机器人相关约束，不进入生成模型。

推荐结构：

```text
lfv/robot/
├── robot_model.py
├── frames.py
├── grasp_to_tcp.py
├── trajectory_conversion.py
├── ik.py
├── collision.py
├── feasibility.py
└── candidate_selector.py
```

主要流程：

1. 将物体坐标系抓取姿态转换到机器人基座坐标系；
2. 根据抓取时的 `T_object_tcp` 将物体运动转换为 TCP 轨迹；
3. 连续求解 IK；
4. 检查环境、自碰撞和夹爪碰撞；
5. 检查关节限位、速度、加速度和轨迹跳变；
6. 对抓取—轨迹组合进行评分；
7. 返回 Top-K 可执行方案。

机器人模块不决定“应该抓哪里”，只判断生成结果能否执行。

### 6.11 `lfv/visualization/`

保存可复用的可视化实现：

```text
lfv/visualization/
├── pointcloud.py
├── contact_field.py
├── gripper.py
├── trajectory.py
├── frames.py
├── open3d_viewer.py
└── video_export.py
```

需要支持：

- RGB 图像上的接触热力；
- 三维点云逐点接触热力；
- 抓取夹爪和左右接触点；
- 物体运动轨迹；
- TCP 和机器人轨迹；
- 坐标轴和坐标变换；
- Open3D 可交互界面；
- 推理和仿真视频导出。

工具脚本应调用这些函数，而不是分别复制一套夹爪绘制代码。

### 6.12 `lfv/utils/`

只保存无明确领域归属的小型基础工具，例如：

- 配置加载；
- 日志；
- 随机种子；
- 文件校验；
- 通用计时器。

几何、扩散、点云或数据 schema 不应因为“很多地方可能使用”就全部放进
`utils/`，应放在对应的领域目录。

## 7. 仿真模块设计

当前仓库已删除旧仿真代码。新模型可以稳定离线推理后，再新增：

```text
lfv/simulation/
├── base.py
├── scene_spec.py
├── observation_adapter.py
├── action_adapter.py
├── rollout.py
├── evaluator.py
├── recording.py
└── backends/
    ├── maniskill.py
    └── mujoco.py
```

各部分职责：

- `base.py`
  - 定义统一 `SimulationBackend` 接口；

- `scene_spec.py`
  - 描述机器人、被操作物体、目标物体、初始位姿和任务成功条件；

- `observation_adapter.py`
  - 将仿真 RGB-D、segmentation 和相机参数转换成与真实数据相同的模型输入；

- `action_adapter.py`
  - 将 TCP 轨迹或关节轨迹转换成具体仿真器动作；

- `rollout.py`
  - 执行 reset、观测、模型推理、规划、动作和终止判断；

- `evaluator.py`
  - 计算抓取成功、任务完成、碰撞和轨迹指标；

- `recording.py`
  - 保存 RGB、深度、点云、状态、动作、视频和评估 JSON；

- `backends/`
  - 隔离 ManiSkill、MuJoCo 等不同 API。

建议统一后端接口：

```python
class SimulationBackend:
    def reset(self, scene_spec): ...
    def get_observation(self): ...
    def get_transforms(self): ...
    def step(self, action): ...
    def check_success(self): ...
    def render(self): ...
    def close(self): ...
```

仿真推理流程：

```text
加载场景
  -> 获取 RGB-D、相机内参和实例分割
  -> 转换为统一物体点云与 DINO 特征
  -> Stage 1 生成接触场和抓取
  -> Stage 2 生成功能轨迹
  -> robot 模块完成坐标转换、IK、碰撞筛选
  -> 仿真器执行预抓取、闭合夹爪和功能运动
  -> 判断任务成功并保存 rollout
```

仿真目录不直接实现网络，也不应依赖训练循环。

## 8. 坐标系契约

整个项目至少涉及：

```text
camera
object_anchor
reference_object
simulation_world
robot_base
tcp
```

每条数据和每次推理必须显式保存：

```text
T_camera_object_anchor
T_world_camera
T_world_robot_base
T_object_tcp
```

推荐统一约定：

- 点云和模型输出使用米；
- rotation 使用右手系；
- 图像反投影使用 OpenCV 相机坐标；
- 模型内部抓取和轨迹使用 `object_anchor` 坐标系；
- 进入 IK 前才转换到 `robot_base`；
- 变换命名统一采用 `T_target_source`，表示把 source 中的点变换到 target。

例如：

```text
p_robot = T_robot_object @ p_object
```

必须为变换方向、旋转正交性和往返变换编写测试。

## 9. 完整数据处理流程

新增一个任务时，按以下顺序完成：

1. 将原始数据放在 `/media/ljian/lj/hand_data/<task>`；
2. 新建 `configs/data/datasets/<dataset>.yaml`；
3. 新建 `configs/data/tasks/<task>.yaml`；
4. 为 GroundingDINO/SAM2 设置被操作物体和参考物体提示词；
5. 运行 episode 准备、分割、点云、跟踪和轨迹恢复；
6. 运行手部检测、接触窗口和接触热力生成；
7. 可选运行 HaMeR 和抓取伪标签；
8. 提取逐点 DINO 特征；
9. 运行质量检查工具；
10. 人工检查少量 good/review/reject 样本；
11. 生成数据集 manifest 和 train/val/test split；
12. Dataset 只读取通过上述处理得到的 artifact。

每个处理阶段必须：

- 有明确输入和输出；
- 保存中间结果；
- 支持 `overwrite=false`；
- 支持单 episode；
- 支持批处理；
- 失败时保存错误原因；
- 不自动删除低质量数据。

## 10. 模型训练流程

### 10.1 Stage 1

建议按三个实验阶段推进：

```text
A. Contact Field overfit
   固定一个小数据集，只训练接触热力

B. Grasp conditional generation
   使用真实 contact_heat 作为条件训练抓取生成

C. End-to-end Contact-Grasp
   使用模型生成的 contact heat 条件训练或联合微调
```

每个阶段都先完成：

- 单 batch 过拟合；
- 单 episode 可视化；
- 小数据集训练；
- 完整数据集训练。

### 10.2 Stage 2

建议顺序：

```text
A. 单轨迹确定性回归基线
B. 单样本 diffusion 过拟合
C. 条件轨迹 diffusion
D. 多模态采样和覆盖率评估
```

必须先确认轨迹坐标系和 normalization 正确，再增加模型复杂度。

### 10.3 通用训练入口

推荐命令形式：

```bash
python scripts/train/train.py \
  experiment=contact_grasp/heat_only \
  data.dataset=hand_pouring_lfv \
  training.output_root=/media/ljian/lj/lfv_checkpoints
```

训练入口应支持：

- resume；
- 单 GPU 和多 GPU；
- AMP；
- EMA；
- 梯度裁剪；
- 固定随机种子；
- train/val 指标；
- 周期性可视化；
- 保存 best 和 latest checkpoint。

## 11. 离线推理和测试流程

离线推理分为三个层次。

### 11.1 单阶段推理

- Stage 1：保存多个 contact/grasp 样本；
- Stage 2：保存多个 functional motion 样本。

### 11.2 两阶段组合推理

组合结果至少保存：

```text
contact_heat_samples
grasp_candidates
motion_trajectory_samples
grasp_motion_pair_scores
selected_plan
coordinate_transforms
quality_report
```

### 11.3 机器人可执行性测试

在不运行完整仿真的情况下，先离线检查：

- 抓取是否位于任务接触热区；
- 夹爪宽度是否可达；
- 夹爪是否与物体或场景碰撞；
- TCP 是否能连续求解 IK；
- 轨迹是否超过速度和关节限制；
- 抓取后物体运动能否转换为连续 TCP 轨迹。

所有推理结果应保存到：

```text
/media/ljian/lj/lfv_model_outputs/<experiment>/<split>/<episode>/
```

## 12. 仿真推理测试流程

仿真测试建议按四个层次推进。

### 12.1 静态场景检查

- 加载机器人和物体；
- 检查模型、碰撞体、单位和坐标轴；
- 检查真实数据与仿真点云方向是否一致；
- 在 UI 中显示接触热力和抓取夹爪。

### 12.2 抓取回放

- 使用给定真值或人工抓取验证 IK 和控制；
- 再使用 Stage 1 生成抓取；
- 只执行预抓取、接近、闭合和抬起；
- 统计抓取成功率和碰撞率。

### 12.3 功能轨迹回放

- 固定一个成功抓取；
- 使用 Stage 2 生成物体轨迹；
- 转换成 TCP 轨迹；
- 检查倒水、拉抽屉等任务完成条件。

### 12.4 完整两阶段闭环

- Stage 1 和 Stage 2 都进行多样本采样；
- 机器人模块选择最优组合；
- 仿真执行完整任务；
- 保存成功率、失败类别、视频和状态日志。

这种分层测试可以区分失败来自：

- 感知和点云；
- 接触区域；
- 抓取姿态；
- 功能运动；
- 坐标转换；
- IK；
- 碰撞；
- 低层控制。

## 13. 可视化要求

每次模型迭代至少提供：

### 数据处理可视化

- RGB、深度和物体 mask；
- 接触窗口；
- 二维接触证据和椭圆热力；
- 三维逐点热力；
- HaMeR 关键点和抓取伪标签；
- 点跟踪和 SE(3) 重投影。

### Stage 1 可视化

- GT 与生成 contact heat 对比；
- 多个接触样本；
- 点云、热力和夹爪的 Open3D 可交互显示；
- 抓取碰撞和接触点。

### Stage 2 可视化

- GT 与生成物体轨迹；
- 多模态轨迹样本；
- 被操作物体和参考物体相对运动；
- 每个时间步坐标轴。

### 仿真可视化

- 场景点云和实例分割；
- 接触热区；
- 候选夹爪和最终夹爪；
- 物体轨迹和 TCP 轨迹；
- IK 失败点和碰撞位置；
- rollout 视频。

可视化输出用于诊断，不应成为训练数据加载的依赖。

## 14. 测试组织

测试目录建议镜像主要代码结构：

```text
tests/
├── data_processing/
├── datasets/
├── models/
│   ├── common/
│   ├── contact_grasp/
│   └── functional_motion/
├── robot/
├── simulation/
└── integration/
```

至少包含：

- 数据字段、shape、dtype 和 NaN 检查；
- 点云与像素对应检查；
- diffusion 加噪和反向采样 shape 测试；
- rotation-6D 转换和 SO(3) 正交性测试；
- SE(3) 变换方向和往返一致性测试；
- 单 batch 训练 smoke test；
- checkpoint 保存和加载一致性；
- 固定随机种子的采样一致性；
- episode_0 端到端离线推理；
- 简化仿真场景的 rollout smoke test。

完整 GPU 训练和长时间仿真不放入普通单元测试。

## 15. 如何添加新模型变体

以新的 Contact Denoiser 为例：

1. 在 `lfv/models/contact_grasp/` 新增实现；
2. 保持输入输出协议与现有 denoiser 一致；
3. 新建 `configs/model/contact_grasp/<variant>.yaml`；
4. 为 shape、mask 和一次 forward 添加单元测试；
5. 使用单 batch overfit 配置验证；
6. 不修改通用 Trainer；
7. 不复制一份新的训练脚本。

如果一个新编码器只用于 Stage 1，先放在 `contact_grasp/`。确认 Stage 2
也使用相同接口和行为后，再提升到 `models/common/`。

## 16. 如何添加新任务

以 drawer opening 为例：

1. 添加数据集配置；
2. 添加任务配置和对象角色；
3. 给出 drawer、handle 等检测提示词；
4. 复用数据处理 pipeline；
5. 对特殊数据格式增加 adapter，不修改模型；
6. 生成与 pouring 相同 schema 的训练 artifact；
7. 为任务增加 train/val/test split；
8. 使用同一 Dataset 类；
9. 通过 experiment config 决定训练单任务还是多任务；
10. 在仿真中添加任务成功判定和场景 spec。

模型不应包含：

```python
if task == "pouring":
    ...
elif task == "drawer":
    ...
```

任务差异应通过数据、条件编码、配置和仿真成功条件表达。

## 17. 模块依赖规则

推荐允许的依赖方向：

```text
scripts
  -> configs
  -> training / inference / evaluation / simulation
  -> datasets / models / robot / visualization
  -> common utilities

data_processing
  -> third_party

datasets
  -> processed artifacts

models
  -> models/common

simulation
  -> inference + robot + visualization
```

禁止的依赖：

- `models` 导入 `training`；
- `models` 导入 `simulation`；
- `datasets` 运行 `data_processing`；
- `data_processing` 依赖训练 checkpoint；
- `robot` 依赖某个具体 diffusion 实现；
- `third_party` 反向导入 LFV；
- `scripts` 承载核心算法。

## 18. 推荐实施顺序

当前最合理的推进顺序为：

1. 固定 processed episode 的数据 schema；
2. 实现 `ContactGraspDataset` 和数据检查；
3. 实现 Stage 1 的 contact heat 最小基线；
4. 完成单 batch 和 episode_0 过拟合；
5. 实现接触条件下的 grasp 生成；
6. 实现 Stage 1 离线推理和 Open3D 可视化；
7. 实现 `FunctionalMotionDataset`；
8. 实现 Stage 2 轨迹 diffusion；
9. 完成两阶段候选组合；
10. 实现机器人坐标转换、IK 和碰撞筛选；
11. 新增统一仿真接口；
12. 先验证抓取，再验证功能轨迹，最后验证完整闭环；
13. 扩展到 pouring 之外的任务。

这个顺序能够让每个阶段都有独立可验证结果，避免数据、模型、机器人和仿真
同时开发时难以定位错误。

## 19. 当前结构需要注意的问题

当前仓库仍处于结构迁移阶段，后续正式实现前应处理：

1. 统一 `contact_grasp_generation` 与 `contact_grasp` 命名；
2. 统一 `functional_motion_generation` 与 `functional_motion` 命名；
3. 避免在 `training/`、`inference/`、`evaluation/` 中复制阶段实现；
4. 明确 `lfv/pipeline/` 向 `lfv/data_processing/` 的渐进迁移方式；
5. 决定任务 YAML 是否为唯一任务定义，避免与空的 `lfv/tasks/` 重复；
6. 从 `pyproject.toml` 中移除已经删除的 `lfv_sim*`；
7. 修正文档中已经迁移到 legacy backup 的旧代码路径；
8. 模型实现前先确定 Dataset schema、坐标系和 normalization。

## 20. 总结

LFV 的结构应围绕三个稳定边界建立：

- 离线数据处理负责从人类视频生成可靠监督；
- 两个生成模型分别负责接触抓取和功能运动；
- robot/simulation 负责把生成结果变成可执行操作。

快速迭代的关键不是创建更多目录，而是保证：

- 一个模型变体主要只修改一个模型目录和一个配置；
- Trainer 不理解具体模型；
- Dataset 不运行重型数据处理；
- 推理不复制模型算法；
- 仿真通过适配器接入，不污染模型；
- 所有中间结果、模型输出和坐标变换都可以保存和检查。

按照本文档组织后，项目可以在保持数据处理成果的基础上，逐步支撑模型训练、
离线推理、机器人可执行性筛选以及多种仿真任务验证。
