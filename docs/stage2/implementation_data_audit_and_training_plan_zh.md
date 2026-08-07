# Stage 2 V2：工程搭建、数据审计与训练实施计划

> 日期：2026-08-07
> 状态：计划已执行；实际代码、训练、推理和执行结果见
> [Stage 2 V2 最终实现文档](final_implementation_pipeline_zh.md)。
> 架构依据：[双对象XYZ–DINO Encoder与三上下文Token设计](unified_scene_goal_trajectory_transformer_design_zh.md)
> 现状依据：[当前GoalPose与Full64计算流程](current_goal_and_trajectory_pipeline_zh.md)

## 1. 目标与本轮边界

本轮要在LFV仓库内部建立可独立训练和推理的Stage 2，不再依赖
/home/users1/ljian/object_centric_diffusion中的历史模型类。第一版完成：

1. 从manipulated/reference两组XYZ+DINO得到三个上下文token；
2. Goal Diffusion生成9D终态位姿；
3. Trajectory Diffusion根据场景上下文和Goal生成64步9D轨迹；
4. 数据审计、离线DINO缓存、训练/验证/测试、EMA和checkpoint闭环；
5. 先通过synthetic与32条真实数据overfit，再进行pouring_lfv完整训练；
6. 保存可复现的K组Goal–Trajectory配对采样及基础可视化。

第一版不加入Contact、环境点云、语言、task token、相对位置注意力、
候选变换几何token、碰撞损失或机器人控制策略。碰撞、IK和执行继续属于后级。

## 2. 已完成的只读初步审查

数据源：

    /media/ljian/lj/data_3d/pouring_lfv

本次审查没有修改数据源。当前得到：

| 项目 | 结果 | 判断 |
|---|---:|---|
| episode总数 | 180 | 数据规模可用于第一版 |
| 完整SE(3)标签episode | 179 | episode_7缺少dp_action_trajectory.npz |
| 现有manipulated二维采样 | 179个episode均为256个不同像素 | 二维索引本身没有64点重复 |
| 现有reference二维采样 | 179个episode均为256个不同像素 | 二维索引本身没有64点重复 |
| 点位全部落在对应mask | 179/179 | mask索引关系正确 |
| 64步轨迹shape/finite/start identity | 179/179 | 基础标签格式正确 |
| 旋转矩阵正交且det约为1 | 179/179 | 基础旋转合法 |
| manipulated现有256点中的有效深度 | min 175，median 221，max 241 | 不能直接当成256个有效三维点 |
| reference现有256点中的有效深度 | min 215，median 230，max 244 | 同样不足256个有效三维点 |
| manipulated mask内可用深度像素 | min 1922 | 足以重新选256个不同点 |
| reference mask内可用深度像素 | min 1750 | 足以重新选256个不同点 |
| Stage 2逐点DINO缓存 | 0/180 | 训练前必须离线生成 |
| object_instance_id元数据 | 0/180 | 暂不能宣称instance-disjoint泛化 |

### 2.1 之前256点训练问题的准确根因

当前文档记录的旧链路有两个不同问题：

1. 历史MultiTaskSE3Dataset内部把num_pts硬编码为64；
2. GoalPoseSE3Dataset再把这64点cyclic padding到256。

所以旧Goal网络看见的256行实际是64个点重复四次。Full64又只使用64点，
并单独计算64点质心，导致Goal和Trajectory阶段的点集与坐标原点不一致。

新的全量审查还发现了第三个问题：源sample_points文件虽然包含256个不同二维
像素，但每个episode只有175--241个manipulated点、215--244个reference点
具有有效深度。旧反投影过程过滤无效深度后仍需要裁剪或补点，因此不能把
“二维索引为256”当成“256个有效且独立的三维点”。

### 2.2 本轮修复原则

不修改原始sample_points文件，也不在数据源目录覆盖历史标签。新缓存构建器：

1. 从首帧完整object mask中取得全部候选像素；
2. 同时过滤有限深度以及0.1--2.0 m工作范围；
3. 在候选中进行确定性图像空间FPS或分布采样；
4. 每个对象选取正好256个不重复像素，禁止replacement和cyclic padding；
5. 用同一像素索引同时反投影XYZ和双线性采样DINO；
6. manipulated/reference都保留256点；
7. Goal和Trajectory共享同一缓存、同一manipulated质心和同一scene scale；
8. 训练时只允许对XYZ与DINO同步做随机置换；
9. 如果某个对象不能提供256个有效不同像素，直接拒绝episode，不补零、不重复。

## 3. 代码目录安排

LFV已有datasets、models、diffusion、training、evaluation和visualization顶层包，
本轮沿用这些边界，不新建第二套平行框架。

~~~text
LFV/
├── lfv/
│   ├── datasets/
│   │   └── functional_motion/
│   │       ├── schema.py
│   │       ├── sampling.py
│   │       ├── cache_builder.py
│   │       ├── dataset.py
│   │       ├── splits.py
│   │       ├── audit.py
│   │       └── synthetic.py
│   ├── models/
│   │   └── functional_motion_generation/
│   │       ├── registry.py
│   │       ├── interfaces.py
│   │       ├── system.py
│   │       ├── encoders/
│   │       │   ├── pointnet.py
│   │       │   └── bidirectional_scene_encoder.py
│   │       ├── blocks/
│   │       │   ├── timestep.py
│   │       │   ├── adaln.py
│   │       │   └── attention.py
│   │       ├── goal/
│   │       │   ├── decoder.py
│   │       │   └── diffuser.py
│   │       └── trajectory/
│   │           ├── decoder.py
│   │           └── diffuser.py
│   ├── diffusion/
│   │   ├── schedulers.py
│   │   ├── normalizer.py
│   │   └── ema.py
│   ├── geometry/
│   │   ├── rotation6d.py
│   │   └── pose9d.py
│   ├── training/
│   │   └── functional_motion/
│   │       ├── trainer.py
│   │       ├── checkpoint.py
│   │       └── logger.py
│   ├── evaluation/
│   │   └── functional_motion/
│   │       ├── metrics.py
│   │       └── evaluator.py
│   └── visualization/
│       └── functional_motion.py
├── configs/
│   └── stage2/
│       ├── base.yaml
│       ├── pouring_lfv.yaml
│       ├── synthetic_overfit.yaml
│       └── real32_overfit.yaml
├── scripts/
│   └── stage2/
│       ├── audit_pouring_lfv.py
│       ├── build_pouring_cache.py
│       ├── train.py
│       ├── evaluate.py
│       └── sample.py
├── tests/
│   └── stage2/
│       ├── test_data_shapes.py
│       ├── test_joint_sampling.py
│       ├── test_rotation6d.py
│       ├── test_schedulers.py
│       ├── test_scene_encoder.py
│       ├── test_goal_diffuser.py
│       ├── test_trajectory_diffuser.py
│       └── test_checkpoint_resume.py
└── docs/stage2/
~~~

当前.gitignore中的datasets/会意外忽略lfv/datasets源码。实施时首先把它改成
仅忽略仓库根目录数据的/datasets/，否则新Dataset代码不会进入Git。

## 4. 处理后数据的安置

源目录保持只读。派生缓存默认放在：

    /home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1

训练输出默认放在：

    /home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1

缓存结构：

~~~text
pouring_lfv_v1/
├── manifest.json
├── audit_report.json
├── split_manifest.json
├── normalizer.json
└── episodes/
    ├── episode_0.npz
    └── ...
~~~

每个episode缓存：

~~~text
manipulated_points       [256,3] float32
manipulated_dino         [256,D] float16/float32
manipulated_pixels_uv    [256,2] int32
reference_points         [256,3] float32
reference_dino           [256,D] float16/float32
reference_pixels_uv      [256,2] int32
goal_pose9d              [9]     float32
trajectory_pose9d        [64,9]  float32
scene_origin             [3]     float32
scene_scale              scalar  float32
episode_id               scalar  string
object_instance_id       scalar  string
source_fingerprint       scalar  string
~~~

scene_origin是首帧manipulated 256点质心。两个对象使用：

    P_local = (P_camera - scene_origin) / scene_scale

轨迹原始变换T=[R,t]转换到这一局部坐标时：

    t_local = (R scene_origin + t - scene_origin) / scene_scale

Goal为trajectory_pose9d最后一帧。rotation6D不做均值方差标准化。

## 5. Dataset职责

FunctionalMotionDataset只读取缓存，不在训练worker中运行DINO、SAM、Zarr解码
或标签构造。返回：

~~~text
manipulated_points [B,256,3]
manipulated_dino   [B,256,D]
reference_points   [B,256,3]
reference_dino     [B,256,D]
goal_pose9d        [B,9]
trajectory_pose9d  [B,64,9]
episode_id
object_instance_id
~~~

Dataset每个epoch可以生成两个独立随机置换，一个作用于manipulated的
points+DINO，一个作用于reference的points+DINO；禁止分别打乱。

训练集统计并保存：

- Goal/trajectory translation各维mean与std；
- scene scale配置与坐标原点定义；
- DINO模型、层、patch size、输入padding和特征维数；
- split manifest和数据源fingerprint。

## 6. Scene Encoder实施

输入只有两组XYZ+DINO。

1. 共享DINO projector：D → 256 → 64；
2. 两个独立XYZ projector：3 → 64；
3. 每点拼接成128维；
4. 两个架构相同但不共享权重的PointNet输出逐点F与全局g；
5. MLP([g_m,g_r])得到z_init；
6. manipulated query reference的一层4-head cross-attention得到z_mr；
7. reference query manipulated的独立cross-attention得到z_rm；
8. 输出三个128维上下文token。

正式接口：

~~~text
ContextEncoding.tokens [B,3,128]
~~~

debug模式额外返回双向attention matrix与两个逐点importance map，但这些不是
Goal或Trajectory输入。

## 7. Goal Diffusion实施

扩散状态为9D：translation3 + rotation6D。采用Hugging Face diffusers：

- DDPMScheduler负责训练add_noise；
- DDIMScheduler负责20步推理；
- 第一版沿用当前已验证的clean-sample prediction，直接预测normalized x0；
- 两个scheduler统一设置prediction_type=sample；
- 100个训练扩散时间步；
- 每个样本随机独立采样t。

Goal Decoder将noisy goal投影为一个128维token，用timestep AdaLN和
cross-attention读取三个context tokens。推荐4层、4 heads。

损失：

    L_goal =
        1.0 * MSE(goal_x0_hat, goal_x0)
      + 1.0 * SmoothL1(t_hat, t_gt)
      + 0.5 * SO3_geodesic(R_hat, R_gt)

物理损失直接由预测x0反归一化得到，不需要再从epsilon解析恢复。

推理从K个不同Gaussian state开始，返回：

~~~text
goals [B,K,9]
goal_id [B,K]
~~~

DDPM的add_noise接口与DDIM的快速迭代step遵循官方文档：

- https://huggingface.co/docs/diffusers/en/api/schedulers/ddpm
- https://huggingface.co/docs/diffusers/en/api/schedulers/ddim

## 8. Trajectory Diffusion实施

扩散状态为第1--63帧的63×9 pose；第0帧固定identity，不加噪。

对每个Goal：

1. MLP(goal)得到一个Goal token；
2. 与三个scene context组成[B,4,128] memory；
3. 63个noisy trajectory pose加连续progress embedding；
4. 每层执行timestep AdaLN、kernel=3 temporal Conv1D、双向时序
   self-attention、trajectory-to-memory cross-attention和FFN；
5. 最终输出63×9 normalized clean trajectory；
6. 拼回固定起点得到64×9轨迹；
7. 最后一帧由模型柔性修正，不硬写成输入Goal。

推荐6层、4 heads、hidden_dim=128。PyTorch attention使用batch_first，
Transformer构造参考官方接口：

- https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

损失：

    L_traj =
        1.0 * diffusion_clean_sample_MSE
      + 1.0 * translation_loss
      + 0.5 * rotation_geodesic
      + 0.2 * velocity_loss
      + 1.0 * endpoint_loss

训练Goal condition按比例混合：

- clean GT Goal；
- 与Goal验证误差匹配的小SE(3)扰动Goal；
- Goal模型达到准入精度后，少量使用其采样结果。

多个Goal必须分别生成轨迹并保留goal_id，不能混在同一条轨迹条件中。

## 9. 统一模型和registry

Stage2System实现：

~~~python
losses = model.compute_loss(batch, stage="goal|trajectory|joint")
samples = model.sample(batch, num_goal_samples=K_goal,
                       num_trajectory_samples=K_traj)
~~~

返回的loss必须分项；samples必须保持Goal–Trajectory配对关系。

registry第一版注册：

    three_token_hierarchical_diffusion

接口保留以后替换：

- deterministic_goal；
- separate_goal_trajectory；
- joint_goal_trajectory_diffusion；
- local_relation_token版本。

## 10. 训练基础设施

Trainer必须包含：

- YAML/Hydra配置；
- Python/NumPy/PyTorch/DataLoader固定seed；
- AMP；
- AdamW；
- gradient clipping；
- EMA；
- best.pt和last.pt；
- 完整断点续训，包括optimizer、scheduler、GradScaler、EMA、epoch和RNG state；
- TensorBoard，W&B可选；
- 训练与验证分项loss；
- 每个epoch固定样本与固定seed的采样快照；
- checkpoint内保存config、normalizer、split、cache fingerprint和模型registry名称。

训练阶段：

1. Phase A：Scene Encoder + Goal Diffuser；
2. Phase B：先冻结Encoder训练Trajectory，再解冻小学习率联合训练；
3. Phase C：Goal与Trajectory联合微调，两个扩散任务独立采样t；
4. 最终使用EMA权重评估和采样。

## 11. 数据深度审计计划

正式audit脚本逐episode记录：

### 11.1 文件和shape

- RGB/depth/mask/trajectory是否存在；
- RGB与depth帧数和分辨率是否一致；
- 两个mask是否非空且与图像同shape；
- 轨迹是否为[64,4,4]和[64,8]；
- episode_7明确列为拒绝项，除非重新生成标签。

### 11.2 新256点合同

- mask内有效深度候选数；
- 选中像素数必须等于256；
- unique pixel必须等于256；
- 反投影后finite点必须等于256；
- unique 3D ratio至少0.99；
- 深度范围、bbox尺寸和点云直径不能异常；
- XYZ和DINO第一维必须完全一致；
- 保存选中pixel hash以检查缓存复现。

### 11.3 DINO

- DINO权重fingerprint；
- 每点feature维数；
- feature必须finite；
- L2 norm分布；
- 重新采样同一像素必须得到一致feature；
- 可视化若干episode的DINO PCA颜色，确认没有源/目标索引错位。

### 11.4 轨迹标签

- start接近identity；
- 所有R满足R转置乘R接近I且det接近1；
- 相邻平移和旋转跳变；
- 终态平移/旋转分布；
- 轨迹总长度与最大单步速度；
- TAPIP3D有效点数；
- 若能从tracking重算，则增加SVD inlier ratio和residual；
- 异常episode进入reject列表，不在Dataset中静默补值。

### 11.5 split

当前180个meta都没有object_instance_id。Dataset与split代码仍强制支持
instance-disjoint，但完整训练前需要选择以下之一：

1. 推荐：补充episode到真实manipulated/reference实例ID的mapping；
2. 仅用于baseline：显式设置allow_episode_id_as_instance=true，使用episode级
   train/val/test split，并在报告中标记不能证明同实例泛化。

默认不偷偷把episode_id冒充真实object_instance_id。

## 12. 单元测试与小数据验证

### 12.1 必须先通过的单元测试

1. Dataset输出shape与dtype；
2. 256个点无重复且XYZ+DINO同步置换；
3. rotation6D ↔ matrix往返、合法性和梯度；
4. DDPM add_noise、sample prediction和已知x0反向step；
5. DDIM单步/完整采样shape和finite；
6. Encoder输入点随机置换后三个token近似不变；
7. 双向attention shape与归一化；
8. Goal forward/loss/sample；
9. Trajectory forward/loss/sample及硬起点；
10. checkpoint保存、恢复、EMA和固定seed采样复现。

### 12.2 Synthetic overfit

- 32条确定性双对象几何样本；
- Goal和trajectory由已知解析关系构造；
- 训练loss必须显著下降；
- 固定样本的终态和轨迹误差同时下降；
- 保存加载后固定seed采样一致。

### 12.3 真实32条overfit

- 从训练split固定选择32条；
- 关闭强augmentation；
- Goal loss、终态平移误差和旋转误差明显下降；
- Trajectory diffusion、平移、旋转、velocity和endpoint loss同时记录；
- 若只能降低训练噪声MSE而物理误差不下降，不进入完整训练。

## 13. 完整训练的准入条件

只有同时满足以下条件才开始pouring_lfv完整训练：

1. 179个有标签episode完成审计，所有拒绝项有明确原因；
2. 每个被接收episode的两个对象都是真实256个有效、不重复三维点；
3. XYZ、DINO、pixel index严格一一对应；
4. Goal与Trajectory读取同一份256点缓存和同一scene origin；
5. split manifest已冻结且无已知instance ID泄漏；
6. DINO离线缓存100%完成并有权重fingerprint；
7. 全部Stage 2单元测试通过；
8. synthetic overfit通过；
9. 真实32条overfit通过；
10. checkpoint resume和固定seed采样复现通过；
11. GPU、diffusers、TensorBoard和磁盘空间检查通过；
12. 完整训练config、seed、run目录和恢复策略写入run manifest。

若缺少真实instance mapping但用户选择baseline模式，可以开始训练，但结果名称
必须包含episode_split_baseline，报告不得使用instance-generalization表述。

## 14. 完整训练计划

第一版建议：

~~~yaml
data:
  num_points: 256
  horizon: 64
  train_val_test: [0.8, 0.1, 0.1]

encoder:
  hidden_dim: 128
  dino_projected_dim: 64
  heads: 4
  cross_layers_each_direction: 1

goal:
  train_diffusion_steps: 100
  inference_steps: 20
  layers: 4
  prediction_type: sample

trajectory:
  train_diffusion_steps: 100
  inference_steps: 20
  layers: 6
  prediction_type: sample

optimization:
  optimizer: AdamW
  learning_rate: 0.0001
  weight_decay: 0.0001
  batch_size: 16
  max_epochs_goal: 300
  max_epochs_trajectory: 500
  max_epochs_joint: 100
  grad_clip_norm: 1.0
  ema_decay: 0.999
  amp: true
~~~

具体epoch由validation early stopping决定，不用固定epoch证明训练完成。best模型：

- Goal：优先val Best-of-K translation + rotation综合指标；
- Trajectory：优先val中间轨迹平移/旋转 + endpoint + velocity综合指标；
- 最终系统：配对Goal–Trajectory的Best-of-K完整指标。

## 15. 最终交付物

完成后应有：

1. 模块化Stage 2源码；
2. pouring_lfv只读审计报告和reject列表；
3. 179条以内的离线XYZ+DINO+pose缓存；
4. frozen split manifest与normalizer；
5. 单元测试和两种overfit报告；
6. Goal、Trajectory和joint的best/last checkpoint；
7. 完整训练TensorBoard/W&B日志；
8. 测试集Top-1与Best-of-K终态/轨迹指标；
9. 固定样本的K组配对Goal–Trajectory可视化；
10. checkpoint恢复与采样复现记录。

## 16. 实施顺序

实际执行严格按以下顺序：

1. 修正.gitignore并建立包和配置骨架；
2. 实现geometry、scheduler、normalizer和schema；
3. 实现审计与cache builder；
4. 生成全量audit报告；
5. 生成离线DINO与256点缓存；
6. 实现Dataset和split；
7. 实现Scene Encoder；
8. 实现Goal Diffuser；
9. 实现Trajectory Diffuser；
10. 实现Trainer、EMA、checkpoint、evaluation和visualization；
11. 运行单元测试；
12. synthetic overfit；
13. 真实32条overfit；
14. 重新执行训练准入检查；
15. 只有准入通过后启动完整训练；
16. 评估、K采样、可视化并更新Stage 2实现文档。
