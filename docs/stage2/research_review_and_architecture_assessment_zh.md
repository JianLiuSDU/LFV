# Stage 2 调研、方法评价与网络结构改进建议

> 审计日期：2026-08-04
> 范围：GoalPose 终态生成、goal-conditioned Full64 中间轨迹、标签构造和 LFV 仿真执行。本文不修改代码，结论来自当前源码、实际 checkpoint resolved config 以及论文原文/作者官方项目。

当前代码的逐步公式、tensor shape、坐标系和 checkpoint 配置见 [Stage 2 当前实现：目标状态生成与中间轨迹生成计算流程](current_goal_and_trajectory_pipeline_zh.md)。

统一 Scene Encoder、Goal Decoder 与目标条件 Trajectory Transformer 的具体 V2 设计见 [Stage 2 V2 架构设计](unified_scene_goal_trajectory_transformer_design_zh.md)。

## 1. 总体判断

当前“两级生成”思想本身是合理的：先建模任务终态分布，再建模满足终态约束的中间 object trajectory，能把“做成什么样”和“怎样到达”分开，也符合 object-centric manipulation 的研究路线。真正的问题不是“完全没有网络”，而是：

1. **现有网络是存在且可描述的**。GoalPose 有 noisy-candidate relation encoder；当前 Full64 有 3 层 Set Transformer 和 1D conditional U-Net。
2. **训练监督构造与推理输入不一致**，有些问题比换网络更优先：GoalPose 训练 64 个点重复成 256，推理却用 256 个真实点；训练输入完整 manipulated mask，推理输入 contact-hot patch；Full64 训练只看 GT goal，部署看预测 goal。
3. **场景信息过早压缩**。Full64 把两组点、起点、目标和语言压成一个 256 维向量，然后 64 个轨迹 token 只能通过相同 FiLM 条件间接接触场景，无法逐时刻查询局部几何。
4. **轨迹状态空间过于通用**。pouring 和 drawer 都被塞进同一个任意 SE(3)×64 表示；drawer 实际是 prismatic joint，最后又靠手写 axis projection 修正，这说明模型表示没有利用任务结构。
5. **生成的是物体运动先验，不是可执行运动规划器**。当前 loss 和 sampling 都不包含机械臂运动学、碰撞、稳定抓取或闭环重规划，不能从轨迹 MSE 推导出执行安全。
6. **工程边界碎片化**。LFV 拥有数据和执行，模型与训练却在历史仓库；基础配置仍是 `todo`，真实接口只能从 checkpoint payload 和适配脚本反推，不利于快速迭代。

因此，下一版不应该继续增加更多手工统计特征。优先方向应是：先修复输入/监督一致性，再建立一个共享的 relational scene encoder，让 goal tokens 和 trajectory tokens 在去噪网络内部直接与 scene tokens 交互；同时对 rigid task 与 articulated task 采用合适的运动流形。

## 2. 对“当前靠繁杂特征提取支撑”的准确判断

这个判断有一半正确、一半需要纠正。

### 2.1 正确的部分

- 离线标签依赖 Grounding DINO、SAM2、深度反投影、TAPIP3D、visibility filtering、两次 SVD 和轨迹重采样，标签链路长，任一误差都会进入监督。
- `cross_attention_encoder.py` 中确实保留了旧 `ManipulationCentricSE3Encoder`：Fourier coordinates、KNN relation、最近区域、质心、协方差、距离统计、max/mean pooling 等多种特征并列，维护成本高，归纳偏置互相重叠。
- 推理又加入 heat quantile crop、score-aware image FPS、256/64 两种质心适配、drawer axis projection，使系统行为分散在很多非学习模块中。

### 2.2 需要纠正的部分

- 当前 pouring 和 drawer 的 Full64 checkpoint **没有使用**上述手工 KNN encoder，而是使用 `GoalConditionedSetTransformerEncoder`。评价当前模型时不能把未启用分支当作现行结构。
- Stage 2 网络**不输入 DINOv2 逐点特征**。Grounding DINO 是 bbox 工具，DINOv2 contact correspondence 属于 Stage 1；Stage 2 的学习输入主要是 raw xyz、goal、start 和一个任务级语言 embedding。
- SAM/TAPIP3D/SVD 是监督标签构造，不等于网络特征工程。视频没有 robot action 或已知物体 pose 时，离线解析不可避免；问题在于当前没有把解析置信度、失败检测和训练权重一起建模。
- GoalPose 不是纯手工回归。它把 noisy candidate cloud 与 reference 做 cross-attention，再进行 diffusion denoising，是一个明确的候选关系评分网络，只是结构仍较弱。

所以更准确的结论是：**当前系统同时存在“离线监督构造过长”和“模型内 scene-to-trajectory 信息瓶颈”，而不是单纯缺少特征。**

## 3. 当前设计值得保留的部分

### 3.1 object-centric SE(3) 中间表示

把视频动作压缩为 manipulated object 相对参考物体的 SE(3) 轨迹，可以跨越 human hand 与 robot end effector 的 embodiment gap。这与 [SPOT](https://arxiv.org/abs/2411.00965) 的核心动机一致：对象轨迹独立于具体机械臂，并能表达倾倒过程中“先移动、后旋转”等中间约束。

### 3.2 终态与路径的概率建模

多种终态或绕行方式不适合普通单峰回归。GoalPose diffusion 与 trajectory diffusion 理论上能保留多模态，而不是平均出无效位姿。这与 [RPDiff](https://arxiv.org/abs/2307.04751) 对多模态 relational pose 的动机以及 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf) 对多模态 action sequence 的动机一致。

### 3.3 局部坐标与关系输入

两组点统一减去 manipulated centroid，标签也围绕同一质心改写，消除了绝对相机平移的一部分影响。Full64 再施加同一 SE(3) augmentation，方向是正确的。

### 3.4 首尾边界 inpainting

把起点和目标作为 hard condition，能让轨迹生成集中学习中段。这是一种清晰、易调试的 baseline；问题不在于使用边界条件，而在于当前没有同时建模边界附近速度、可行性和 predicted-goal uncertainty。

### 3.5 object-to-TCP 固定 attachment

稳定抓取后，用固定 $T_{object\rightarrow tcp}$ 把 object trajectory 变成 TCP trajectory，模块职责清晰，适合第一版验证。它需要闭环 slip monitoring，但不应因为存在误差就完全取消 object-centric 设计。

## 4. 优先级最高的实现与数据问题

### P0-1：GoalPose 的“256 点”训练接口与真实内容不一致

`MultiTaskSE3Dataset` 将点数硬编码为 64，`GoalPoseSE3Dataset` 再 cyclic padding 到 256。也就是说训练中的 attention 看见每个几何点四次；仿真推理则用 256 个真实、空间分布不同的点。

影响：

- attention 密度和重复点权重发生变化；
- 训练时的 object geometry 分辨率低于配置表面含义；
- 不能用“GoalPose 使用 256 点”解释模型容量或泛化。

这应在任何架构替换之前修复并做同分布对照实验。

### P0-2：训练看完整 manipulated part，推理看 contact-hot patch

pouring 训练的 manipulated cloud 来自整只 cup mask，drawer 训练来自整个 handle mask；当前推理却只在 contact heat 的高分 quantile 内取点。对于杯子，这可能只剩把手局部；对于抽屉，可能只剩把手中心。

GoalPose 的目标是学习 manipulated object 与 reference 的终态关系，它需要物体尺度、朝向和外形。把 contact 区域当作整个 manipulated object，会改变 centroid、bbox scale、旋转可观测性和与 reference 的距离关系。

推荐的语义是：

- 完整 manipulated point set 始终作为几何主体；
- contact heat 作为每点一个 soft channel 或额外 contact tokens；
- 不能通过裁掉低热点来表达“哪些点重要”。

### P0-3：Full64 训练只看 GT goal，部署只看 predicted goal

实际 checkpoint 配置中：

```yaml
goal_noise_std_xyz: 0.0
goal_noise_std_rot: 0.0
```

而 `_normalize_goal_obs` 代码只实现了 xyz noise，`goal_noise_std_rot` 即使非零也未被使用。Full64 因此没有学习如何处理 GoalPose 的位置和旋转误差，形成标准的 teacher-forcing/exposure gap。

需要在训练中混入：

- GoalPose checkpoint 的真实采样结果；或
- 与 GoalPose validation error 分布一致的 SE(3) 扰动；或
- scheduled conditioning，从 GT goal 逐步过渡到 predicted goal。

评价时必须同时报告 `GT-goal -> trajectory` 和 `predicted-goal -> trajectory`，否则不能定位是第一级还是第二级失败。

### P0-4：episode split 不能证明实例泛化

当前按 episode 随机 90/10 划分，没有读取或约束 `object_instance_id`。如果多个 episode 使用同一杯子/抽屉实例，验证集可能只是在测同实例、相近视角和相近场景的重放能力。

需要建立：

- instance-disjoint split；
- scene/view-disjoint split；
- task内 in-distribution split；
- unseen-category 或 unseen-articulation split（若目标包含跨类别）。

### P0-5：drawer 被错误地当成自由 SE(3) 任务

drawer handle 的合法运动由一个 prismatic joint 决定，本质自由度接近一个标量 $s_t$：

$$p_t=p_0+s_ta,\qquad R_t=R_0.$$

当前模型却生成 64 个任意 7D pose；执行时再用 `project_prismatic_trajectory` 丢弃横向位移和全部旋转，强制单调。这表示主要结构约束没有被学习，验证的 SE(3) MSE 也包含最终会被删除的自由度。

更重要的是，原始 [SPOT](https://arxiv.org/html/2411.00965v2) 实验明确把 articulated objects 排除在范围外。当前 drawer 扩展不是简单换一个任务配置，而是改变了任务的运动学类别。

### P0-6：当前不是闭环轨迹生成

原始 SPOT 在执行中跟踪 object pose，并以 receding horizon 反复生成未来轨迹。当前 LFV 只在初始 snapshot 上运行一次 GoalPose 和一次 Full64，随后执行 64 步固定 TCP subgoals。

这会放大：

- grasp attachment 误差；
- 夹爪内滑动；
- Panda tracking error；
- drawer 摩擦和关节约束导致的实际/预测偏离。

应把“网络预测质量”和“开环累计误差”分开评价。

## 5. 网络结构上的主要问题

### 5.1 GoalPose：有网络，但关系推理仍过度全局化

当前 GoalPose 的优点是显式构造 noisy candidate cloud，使 denoiser 能问“这个候选变换后是否符合 reference relation”。但问题包括：

- 每个 relation branch 只有一次 manipulated-to-reference cross-attention；没有反向 cross-attention、局部邻域或多尺度对应。
- max/mean pool 后只保留单一全局向量，精确接触、容器开口、把手轴线等局部关系可能消失。
- static/candidate branch 权重不共享，两个分支学到的 relation space 没有结构性一致保证。
- 原始 xyz MLP + bbox scale normalization 不具备严格 SE(3)/SIM(3) equivariance。
- 在普通 9D 欧氏空间扩散，旋转由 6D projection 回 SO(3)；可用但没有在流形上定义噪声。
- 当前命名为 `RPDiffStyle`，但不是 RPDiff 原架构的复现。RPDiff 强调相关局部几何和精细候选去噪；这里是更轻量的全局 relation baseline。

可对照两类目标模型：

- [TAX-Pose](https://arxiv.org/abs/2211.09325) 学习双向 soft cross-object correspondence 和 point weights，再用 differentiable weighted SVD 得到 task-specific cross-pose；它提供精确、可解释的确定性终态 baseline。
- [RPDiff](https://arxiv.org/abs/2307.04751) 直接对 relational pose 做迭代去噪，适合一个场景存在多个几何可行终态的情况。

合理路线不是盲目二选一：先建立 TAX-Pose-like deterministic correspondence baseline，确认数据中终态关系可学；需要多模态时，再用 residual/pose diffusion 扩展。

### 5.2 Full64：Set Transformer 是合理起点，但 256D global bottleneck 太强

当前 scene encoder 先得到 37 个 token，经过 Transformer 后只取 CLS，压为 256 维；temporal U-Net 的每个时间位置只能通过相同 FiLM 参数使用它。

这会导致：

- 第 8 步和第 56 步无法分别查询不同的 scene region；
- noisy trajectory token 不能直接 attend 到 manipulated/reference point tokens；
- goal condition 与 local geometry 的交互只发生在压缩前一次；
- 细长把手、碗口、杯沿等小结构容易在 pooling 中损失。

更合适的是把 `[B,T,d]` trajectory tokens 保留在 denoiser 中，并在每层做：

```text
trajectory self-attention
  -> trajectory-to-scene cross-attention
  -> goal/timestep AdaLN or FiLM
  -> feed-forward
```

[ManiFlow](https://arxiv.org/abs/2509.01819) 的 DiT-X 使用 action tokens 与多模态 observation 的 adaptive cross-attention，说明“保留 token 级交互”是比单个 global vector 更清晰的现代结构方向；这里引用的是架构启发，不意味着应直接复制其规模或训练配方。

### 5.3 trajectory state 使用 quaternion + 欧氏 MSE 不够严谨

当前 action `[t,q_xyzw]` 的平移按数据范围归一化，四元数只单位化，然后对 7 个维度等权加高斯噪声和 MSE。问题是：

- $q$ 与 $-q$ 表示同一旋转，虽在标签中做连续性修正，扩散中间态仍不在 $S^3$；
- translation 与 quaternion component 的数值误差没有直接物理可比性；
- 训练 loss 没有 SO(3) geodesic、transformed-point 或相邻运动约束；
- 最后归一化 quaternion 只能恢复合法旋转，不能保证轨迹旋转平滑。

候选表示包括：

- 每步 translation + 6D continuous rotation；
- 相对起点或相邻步的 SE(3) Lie algebra twist；
- 少量 B-spline/control poses，再连续插值为执行频率。

[Motion Planning Diffusion](https://arxiv.org/abs/2412.19948) 使用 B-spline motion primitives 压缩轨迹，并在 sampling 时加入 cost guidance，直接对应当前“64 个高冗余 waypoint + 无 smoothness/collision”的问题。

### 5.4 hard goal boundary 隐藏了终点问题

第 63 帧由 predicted goal 强制写入，所以 Full64 的 endpoint error 对该 goal 条件恒为零；它不能证明最后一段路径能平滑、无碰撞地到达终点。当前 best checkpoint 选择中间平移误差比选全轨迹 MSE合理，但还缺少：

- 倒数若干步的 velocity/acceleration continuity；
- 最后一段 swept collision；
- 对 predicted-goal perturbation 的条件稳定性；
- Best-of-K coverage 与 sample diversity。

### 5.5 goal token 存在冗余，语言条件在单任务训练中不可辨识

数据起点通常是 identity，因此 `goal_pose9d` 与 `goal_delta_pose9d` 几乎相同，却被作为两个独立 token。pouring/drawer 又使用各自独立 checkpoint，每个训练集所有 episode 共用同一个 task embedding，因此语言 token 不能通过数据学出任务区分，更多像固定 bias。

在单任务 baseline 中应做 ablation：删除 absolute goal 或 delta goal 中一个、删除 language token。只有多任务联合训练时，语言条件才有可验证意义。

## 6. 标签与损失结构的问题

### 6.1 TAPIP3D + SVD 是可用标签器，但置信度没有传到训练

当前 SVD 用可见性加权做第一次拟合，再以残差阈值做等权 refit，这是合理 robust baseline。但最终训练只保留 pose，不保留：

- 每帧有效 tracks 数量；
- SVD residual；
- inlier ratio；
- rotation condition number；
- occlusion gap；
- episode-level label confidence。

结果是低质量标签与高质量标签等权。建议先把这些质量量作为数据审计/采样权重，而不是再添加新的视觉 descriptor。

### 6.2 刚体 SVD 的适用边界

- cup 作为刚体，SVD 压缩合理；但单视角、手遮挡和水杯对称性会使旋转不稳定。
- drawer handle 与 drawer front 刚性相连，跟踪 handle 可恢复 drawer motion；但它的运动必须满足 cabinet joint，而不是任意 SVD SE(3)。
- 如果以后处理布料、液体表面或非刚体物体，单一 SE(3) 会系统性丢失任务信息，应改为 point/scene flow 或关键点轨迹。

### 6.3 重采样与采样实现存在维护信号

- LFV 标签器先按 translation+rotation arc length 重采样到 64。
- Full64 dataset 又连续调用了两次 `resample_se3_trajectory`；因为第一次已返回 64 点，第二次目前无数值作用，但属于重复代码。
- GoalPose 底层 64 点再复制为 256。
- 推理 Full64 直接取 GoalPose 256 点序列的前 64 点，而不是显式运行与训练一致的 sampler。

这些问题说明 point/trajectory sampling contract 没有成为单一、可测试的基础设施。

## 7. 为什么当前输出不能自然保证碰撞与可执行性

Full64 预测的是 object-relative pose，训练条件只有 manipulated/reference 局部点集。它没有看到：

- 完整桌面、cabinet、robot links 和其他障碍物；
- 关节限位、IK 可达性和奇异位形；
- gripper finger swept volume；
- 当前抓取的真实接触与 slip；
- drawer joint dynamics。

因此“数据中的人类轨迹没有碰撞”只能提供经验先验，不能保证新场景安全。当前执行脚本中的 `collision-checked preshaped approach` 名称主要继承了 GraspNet 抓取候选阶段的碰撞筛选；Full64 后续插值并没有逐步 collision planning。

可以参考两条互补路线：

- [StructDiffusion](https://arxiv.org/abs/2211.04604) 在目标结构生成之外使用物理有效性判别器筛掉不合格结果，说明“生成分布”和“物理有效性”可以分模块处理。
- [Motion Planning Diffusion](https://arxiv.org/abs/2412.19948) 把 learned trajectory prior 与环境 cost gradient 结合，说明可以在去噪时显式加入碰撞/平滑代价。

对 LFV 更现实的第一步是：object trajectory model 负责任务运动先验；机器人层用 IK、collision checker 和 trajectory optimizer 做可执行化，而不是要求一个 82/106 episode 的网络隐式学会全部机器人约束。

## 8. 点流/光流路线与当前方法的关系

### 8.1 点流为什么与当前数据天然兼容

当前标签本来就是：TAPIP3D 产生逐点 3D tracks，再用 SVD 把它们压成 SE(3)。因此可以把监督保留在压缩前，学习：

$$F_{k,i}=p_{k,i}-p_{0,i}\in\mathbb R^3,$$

再用 differentiable weighted Procrustes/SVD 恢复每一步 $T_k$。这会保留局部空间结构，并允许网络显式关注高置信 contact/handle points。

[ToolFlowNet](https://sites.google.com/view/point-cloud-policy) 正是以 PointNet++ 输出工具逐点 flow，再通过 differentiable SVD 得到工具变换；其 pouring 任务与这里非常接近。[TAX-Pose](https://arxiv.org/html/2211.09325v3) 也展示了“学习对应/flow，解析求 SE(3)”的可解释结构。

### 8.2 articulated drawer 更适合 articulation flow

[FlowBot3D](https://arxiv.org/abs/2205.04382) 为 articulated object 预测逐点 articulation flow，并用解析 planner 执行动作。对 drawer，flow 的方向自然沿 prismatic axis，比任意 SE(3)×64 更贴合物理结构。

但 LFV 已经有完整仿真 joint 信息时，不必用网络重新发现所有运动学。可用两级方案：

- sim/known asset：直接使用已知 prismatic axis 和预测标量 progress；
- real/unknown asset：用 articulation-flow/axis head 估计轴，再生成 constrained progress trajectory。

### 8.3 2D point tracks/optical flow 适合视频迁移，不等于可执行轨迹

- [ATM](https://arxiv.org/abs/2401.00025) 预测任意图像点未来轨迹，并把它作为下游 policy 的控制指导。
- [Im2Flow2Act](https://arxiv.org/abs/2407.15208) 将语言条件 object flow 作为 human video 与 simulated robot policy 之间的接口，明确分成 flow generator 和 flow-conditioned policy。

这些工作支持“flow 是跨 embodiment 的中间表示”，但也说明 flow 本身不是 robot action。仅从 2D optical flow 用 SVD 得 6DoF 会受深度、遮挡、aperture ambiguity 和离平面运动影响；LFV 已有 RGB-D/TAPIP3D，优先使用 3D tracks/point flow 更合理。

### 8.4 不建议完全抛弃 SE(3)

[SPOT](https://arxiv.org/html/2411.00965v2) 明确指出 dense point flow 可能冗余且含噪，直接 object pose 更紧凑。这个判断对 rigid cup 是合理的。因此推荐：

- rigid pouring：SE(3) trajectory 作为主输出，sparse point-flow consistency 作为辅助或可解释中间层；
- articulated drawer：joint progress/articulation flow 作为主输出；
- deformable tasks：point trajectory/flow 作为主输出，不再强行 SVD 成单一刚体。

## 9. 与相关工作的结构对照

| 工作 | 核心表示/结构 | 对当前系统的启发 | 不能直接照搬之处 |
|---|---|---|---|
| [SPOT](https://arxiv.org/abs/2411.00965) | target-frame object SE(3) trajectory diffusion + online tracking + receding horizon | 保留 object-centric trajectory；恢复闭环 | 原论文不处理 articulated object，且依赖 6D pose tracking/mesh |
| [RPDiff](https://arxiv.org/abs/2307.04751) | 局部关系几何上的多模态 pose denoising | 改进 GoalPose 的局部关系与多样性 | 主要解决终态 rearrangement，不生成完整 robot-feasible path |
| [TAX-Pose](https://arxiv.org/abs/2211.09325) | 双向对应、权重、residual correspondence、weighted SVD | 终态的可解释强 baseline；保留点级结构 | 确定性版本对多模态终态不足 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf) | 条件 action-sequence diffusion | 说明 1D temporal U-Net baseline 合理 | 原方法面向 robot action，不自动提供 SE(3) equivariance/碰撞保证 |
| [DP3](https://arxiv.org/abs/2403.03954) | sparse point cloud + 简洁 point encoder + diffusion policy | 少量干净 3D 表示可胜过复杂特征堆叠 | 当前任务是 object motion 且有双对象关系，不应只复制单全局点云 encoder |
| [EquiBot](https://arxiv.org/abs/2407.01479) | SIM(3)-equivariant point encoder 与 equivariant diffusion U-Net | 减少相机/位置/尺度增强负担 | 工程复杂度更高；scale equivariance 对固定资产任务未必第一优先 |
| [ToolFlowNet](https://sites.google.com/view/point-cloud-policy) | per-point tool flow + differentiable SVD | 与当前 TAPIP3D→SVD 标签天然对齐；尤其适合 pouring | 单步/闭环 policy 设定与离线 Full64 不完全相同 |
| [FlowBot3D](https://arxiv.org/abs/2205.04382) | dense articulation flow + analytical planner | drawer 应使用 articulation manifold | 面向最大化 articulation，不直接表达演示风格的完整 64 步路径 |
| [ATM](https://arxiv.org/abs/2401.00025) | 2D any-point future tracks + downstream policy | 可利用大量 action-free video | 2D tracks 缺少直接 3D 可执行性 |
| [Im2Flow2Act](https://arxiv.org/abs/2407.15208) | language-conditioned object flow + simulated flow-conditioned policy | 明确 human-video representation 与 robot policy 的接口 | 需要额外 simulated play/action data，不是只换一个 decoder |
| [Motion Planning Diffusion](https://arxiv.org/abs/2412.19948) | B-spline trajectory latent + cost-guided posterior sampling | 低维、平滑、碰撞 guidance | 面向 motion planning prior，不能替代任务语义目标生成 |

## 10. 推荐的 Stage 2 V2 计算结构

以下是建议的目标结构，不是本次代码修改。

### 10.1 最小双对象 relational scene encoder

2026-08-07重审后，Stage 2的最小输入contract缩减为：

```text
manipulated_points [B,Nm,3]
manipulated_dino   [B,Nm,D]
reference_points   [B,Nr,3]
reference_dino     [B,Nr,D]
```

这四个张量已经包含当前Stage 2需要的对象几何与功能语义。Contact仍用于Stage 1接触迁移和抓取，不进入Stage 2；environment、task type和language也不进入单任务checkpoint。这个边界成立的前提是：manipulated/reference mask由数据管线给出，pouring与drawer分别训练，并由生成后的碰撞检查、IK和执行层处理环境可行性。

编码流程同步缩减为：

1. 两个对象使用共享的低维DINO projector，使语义仍位于同一特征空间；
2. XYZ分别投影后与逐点DINO拼接，送入架构相同、权重独立的两个轻量PointNet；
3. 合并两个PointNet全局特征，得到初始双对象上下文$z_{init}$；
4. manipulated逐点特征query reference逐点特征，得到$z_{m\leftarrow r}$；
5. reference逐点特征反向query manipulated逐点特征，得到$z_{r\leftarrow m}$；
6. 最终只输出三个固定职责的上下文token：

```text
Z_ctx = [z_init, z_manipulated<-reference, z_reference<-manipulated]
      in R^[B,3,128]
```

双向attention map仅作为诊断可视化，不扩大正式模型接口。三个全局token是有意设置的强信息瓶颈：适合先验证当前小数据单任务，但不能假定对所有精细局部关系都无损。若attention已经定位正确而终态仍不精确，下一步应给每个方向保留少量4--8个relation tokens，而不是恢复Contact、环境或手工特征堆叠。完整计算与边界见[V2详细设计](unified_scene_goal_trajectory_transformer_design_zh.md)。

### 10.2 目标状态生成

对于rigid relation task，Goal Decoder直接以三个上下文token为memory：

```text
3 context tokens + noisy 9D goal token + diffusion timestep
  -> timestep AdaLN + goal-to-context cross-attention
  -> 预测 SE(3) denoising update
  -> K 个候选 goal
```

最小版不构造candidate-transformed geometry token或3D relative bias。9D状态由translation3与rotation6D组成；条件信息不是9维，而是三个128维上下文token，因此低维扩散状态本身不会妨碍网络读取场景关系。碰撞与可达性仍在生成后评分。

推荐先实现两个可比较 baseline：

1. **Deterministic correspondence baseline**：TAX-Pose-like soft correspondence + weighted SVD；
2. **Multi-modal goal diffusion**：RPDiff-like relation denoiser，输出 Best-of-K。

对于 drawer：目标不应输出任意 SE(3)，而应输出 desired joint progress $s_{goal}$，可选同时估计 axis/confidence；已知仿真资产时 axis 是观测条件，不是学习标签。

### 10.3 中间轨迹生成

推荐用trajectory-token DiT替换“scene CLS -> global FiLM only”。对每个Goal候选先得到一个goal token，再与三个Encoder token组成四token memory：

```text
noisy trajectory tokens [B,T,d]
hard start + soft goal condition + diffusion timestep
memory = [3 context tokens, 1 goal token] [B,4,d]

for each block:
  local temporal Conv1D
  temporal self-attention
  cross-attention(query=trajectory, key/value=memory)
  timestep AdaLN
  FFN
```

2026-08-06重审后，推荐只hard-inpaint起点，最后一帧与中间轨迹一起去噪；Trajectory head用匹配Goal验证误差的小扰动训练，并把最后一帧作为`refined goal`。多个Goal候选按`goal_id`并行生成配对轨迹，而不是无标识地同时输入一条轨迹。

输出表示按任务类型选择：

- rigid task：`translation + rotation6d` 的 object pose/control poses；
- drawer：单调 prismatic progress sequence $s_{1:T}$；
- 可选 sparse point-flow head：预测若干 object anchors 的 $F_{t,i}$，经 weighted SVD 与 pose trajectory 做 consistency。

Full64 不一定必须直接生成 64 个独立 waypoint。可先生成 8–16 个 control poses，再用 SE(3) interpolation/B-spline 展开到控制频率，以减少冗余和抖动。

### 10.4 goal uncertainty 传播

训练 Trajectory head 时混合三种条件：

```text
GT goal
perturbed GT goal (匹配 GoalPose val error)
frozen/current GoalPose sampled goal
```

若 GoalPose 输出 K 个候选，Trajectory head 对每个候选生成若干路径，再联合以 relation、collision、reachability 和 trajectory likelihood 排序。不能先只选一个位置最近的 goal，再假设后级能修正。

### 10.5 可执行性层

建议明确三层职责：

```text
task motion prior       : object trajectory model
geometric feasibility  : full-scene collision/SDF + IK/reachability
feedback execution     : object/TCP tracking + receding horizon
```

最小版本可在生成后进行 collision-aware trajectory optimization；后续再把 differentiable collision cost 加入 denoising guidance。drawer 直接在 prismatic manifold 生成，不再先错后投影。

## 11. 推荐的验证与消融顺序

在更换大网络前，建议依次完成以下实验，否则无法知道收益来自哪里。

### 第一组：修复数据 contract

1. GoalPose 64 unique vs 256 unique，禁止 64×4 重复；
2. complete manipulated cloud vs hot-region-only cloud；
3. `xyz only` vs `DINO only` vs `xyz+DINO`；
4. instance-disjoint split；
5. 记录 label inlier ratio/residual，并比较过滤前后。

### 第二组：目标模型

1. 简单 MLP/global pooling；
2. 当前 GoalPose；
3. TAX-Pose-like correspondence baseline；
4. token-level relational goal diffuser；
5. 报告 top-1/Best-of-K translation、SO(3)、relation success、collision 和 diversity。

### 第三组：轨迹模型

1. GT-goal + 当前 Full64；
2. predicted-goal + 当前 Full64；
3. predicted-goal-noise training；
4. global-FiLM U-Net vs trajectory-scene cross-attention DiT；
5. 64 waypoints vs control poses/B-spline；
6. pose-only vs pose + sparse flow consistency。

### 第四组：任务与执行

- pouring：upright-before-pour、pour position/orientation、路径碰撞、rollout success；
- drawer：axis angular error、orthogonal drift、monotonicity、joint progress error、实际拉出距离；
- 两者：open-loop vs receding horizon、grasp slip、IK failure、swept collision、最终任务成功率。

## 12. 工程结构评价

### 12.1 当前问题

- [functional_motion/base.yaml](../../configs/experiments/functional_motion/base.yaml) 仍写 `object_encoder: todo`，与真实 checkpoint 架构矛盾。
- 训练核心在 `/home/users1/ljian/object_centric_diffusion`，LFV 通过 `sys.path`、`os.chdir`、Hydra checkpoint payload 动态加载；无法在 LFV 内独立训练、测试或替换模型。
- `simple_dp3.py` 同时包含三个 encoder 分支、大量失效注释、unused `noise_scheduler_pc`、未实现的 rotation goal noise 和被强制关闭的 progress 分支。
- `two_stage_pouring.py` 已承担 task-neutral 逻辑，但文件名仍绑定 pouring；另有通用和 pouring 专用推理脚本并存。
- 当前 LFV 测试覆盖坐标转换和 drawer projection，但不覆盖 GoalPose/Full64 forward、normalizer、训练—推理 point contract、predicted-goal robustness 或 checkpoint reproducibility。
- GoalPose 与 Full64 没有统一的 LFV model registry/interface；采样 K、返回 uncertainty、metrics schema 也不统一。

### 12.2 推荐的最终模块边界

```text
lfv/stage2/
  data/
    trajectory_label_builder.py
    episode_dataset.py
    sampling_contract.py
  geometry/
    pose_repr.py
    weighted_procrustes.py
    trajectory_parameterization.py
  models/
    registry.py
    scene_encoder.py
    goal_generators/
    trajectory_generators/
  diffusion/
    schedulers.py
    boundary_conditioning.py
  inference/
    hierarchical_sampler.py
    feasibility_filter.py
  evaluation/
    goal_metrics.py
    trajectory_metrics.py
    rollout_metrics.py
  visualization/
  configs/
  tests/
```

统一接口可定义为：

```python
goal_samples = goal_model.sample(scene_batch, num_samples=K_goal)
trajectory_samples = trajectory_model.sample(
    scene_batch,
    goal_samples,
    num_samples=K_traj,
)
losses = model.compute_loss(batch)
```

checkpoint 内必须保存完整 model config、point sampling contract、normalizers、pose representation、dataset split manifest 和 label schema。LFV 只依赖这个稳定接口，不再知道历史训练仓库的内部类名。

## 13. 最终建议

当前最有价值的资产不是某个复杂 encoder，而是已经跑通的 object-centric 数据闭环：视频点跟踪、SVD 轨迹标签、GoalPose/Full64 分级生成、抓取 attachment 和仿真执行。应该保留这一任务分解，但重做其信息流：

1. 先消除 64→256 重复、完整物体→热区裁剪和 GT goal→predicted goal 三个分布断点；
2. 用共享 scene tokens 取代多处分散的特征/适配逻辑；
3. 让 trajectory tokens 在去噪过程中直接 cross-attend scene tokens；
4. rigid pouring 使用 SE(3) 或 pose+flow consistency，drawer 使用 prismatic/articulation 表示；
5. 恢复 receding-horizon feedback，并把碰撞/IK 作为显式 feasibility layer；
6. 最后再比较 equivariant encoder、flow matching 或更大 DiT，而不是先扩大网络。

这条路线既能回应“结构设计不清晰”的问题，也能最大程度复用现有标签、仿真和执行基础设施。
