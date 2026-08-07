# Stage 2 当前实现：目标状态生成与中间轨迹生成计算流程

> 审计日期：2026-08-04
> 本文描述当前实际用于 pouring 与 drawer checkpoint 的代码，而不是 `configs/experiments/functional_motion/base.yaml` 中仍标为 `todo` 的早期设计。本文只记录现状，不提出代码修改。

问题分级、论文对照和下一版结构建议见 [Stage 2 调研、方法评价与网络结构改进建议](research_review_and_architecture_assessment_zh.md)。

## 1. 系统边界与当前结论

当前 Stage 2 是一个两级生成系统：

1. **GoalPose** 根据初始 manipulated/reference 点云和任务语言，生成 manipulated object 相对初始状态的终态 SE(3) 位姿。
2. **Full64** 根据同一初始几何、起点和 GoalPose 结果，生成从单位起点到该终态的 64 个 object-centric SE(3) 位姿。
3. LFV 把 object trajectory 与抓取时的固定 `object -> TCP` 变换组合成 TCP trajectory，再交给 ManiSkill 的 Panda 控制器执行。

这不是一个端到端 LFV 模型。职责目前分布在两个仓库中：

- `LFV`：RGB-D 数据预处理、TAPIP3D 轨迹标签、仿真输入适配、两阶段串联、坐标转换、可视化和执行。
- `/home/users1/ljian/object_centric_diffusion`：GoalPose 与 Full64 数据集、网络、训练 workspace、normalizer 和 checkpoint。

当前生效的网络关系可概括为：

```text
训练视频 RGB-D
  -> manipulated/reference 首帧 mask 与采样点
  -> TAPIP3D 跟踪 manipulated 点
  -> visibility-weighted SVD
  -> 64 步 SE(3) 标签
  -> GoalPoseDiffuser:  初始双点云 -> 终态 9D pose
  -> Full64 Diffuser: 初始双点云 + 终态 -> 64 x 7D pose trajectory

仿真单帧 RGB-D + Stage 1 contact heat
  -> heat-aware manipulated 点 + reference mask 点
  -> GoalPose 采样
  -> Full64 采样
  -> camera-local delta -> world object pose
  -> 固定 grasp attachment -> world TCP pose
  -> 开环执行整条轨迹
```

## 2. 当前生效的数据与 checkpoint

| 任务 | Stage 2 训练目录 | 有效 episode | Goal checkpoint | Full64 checkpoint |
|---|---:|---:|---|---|
| pouring | `/media/ljian/lj/data_3d/pouring` | 82 | `pouring_seed42/.../epoch=0700-val_sample_goal_pos_err_cm=3.086.ckpt` | `pouring_seed42/.../epoch=1500.ckpt` |
| drawer | `/media/ljian/lj/data_3d/drawer_lfv_v2_train` | 106 个指向 `drawer_lfv_v2` 的软链接 | `drawer_lfv_v2_train_seed42/.../epoch=0800-val_sample_goal_pos_err_cm=2.115.ckpt` | `drawer_lfv_v2_train_seed42/.../epoch=1400.ckpt` |

两个数据集都用固定随机顺序按 episode 做 90%/10% 切分：pouring 为 73/9，drawer 为 95/11。这个切分是 episode 级，不是由 `object_instance_id` 保证的实例级切分。

当前 checkpoint 的共同关键配置为：

- GoalPose：256 点输入接口、9D 终态、100 个训练扩散步、10 个 DDIM 推理步、`prediction_type=sample`。
- Full64：64 点输入、64 个轨迹位姿、Goal-Conditioned Set Transformer、1D conditional U-Net、首尾边界 inpainting、100 个训练扩散步、10 个 DDIM 推理步、`prediction_type=sample`。
- 两个任务各自单独训练，均加载各任务共享的一个 1024 维语言 embedding。

## 3. 轨迹监督标签如何生成

### 3.1 首帧物体与参考物体

数据处理配置位于：

- pouring：[picknplace.yaml](../../configs/pipeline/picknplace.yaml)
- drawer：[drawer_motion.yaml](../../configs/pipeline/drawer_motion.yaml)

对于首帧 RGB：

1. Grounding DINO 根据文本提示产生 manipulated object 和 reference object 的 bbox。
2. SAM2 根据 bbox 产生两个二值 mask。
3. [sample_points.py](../../lfv/pipeline/sample_points.py) 在每个 mask 的 bbox 中构造均匀图像网格，再随机裁剪或重复填充到配置点数。
4. 使用首帧深度与相机内参反投影：

   $$
   p(u,v)=D(u,v)
   \begin{bmatrix}
   (u-c_x)/f_x\\
   (v-c_y)/f_y\\
   1
   \end{bmatrix}.
   $$

pouring 中 manipulated 是整只 cup，reference 是 bowl；drawer 中 manipulated 是黑色 drawer handle，reference 是排除了活动抽屉面板与把手的 cabinet housing。

### 3.2 TAPIP3D 只跟踪 manipulated 点

[tracking.py](../../lfv/pipeline/tracking.py) 按 `tapip3d.sample_candidates` 顺序加载第一份存在的点文件。当前配置里 manipulated 点路径排在 reference 点路径之前，所以正常情况下 TAPIP3D 只跟踪 manipulated 点；reference 路径只是 manipulated 文件缺失时的 fallback，并不是第二组同时跟踪的点。

首帧 sampled pixels 由深度提升为相机坐标系三维 query points，TAPIP3D 输出：

- `coords[t,i,3]`：第 $i$ 个点在第 $t$ 帧的三维相机坐标；
- `visibs[t,i]`：该点可见性/置信度。

相机外参序列在这一步被设为单位阵，因此标签表达的是相机坐标系中的观测运动。

### 3.3 visibility-weighted SVD 压缩为刚体 SE(3)

[se3_trajectory.py](../../lfv/pipeline/se3_trajectory.py) 以首帧点 $P_i$ 为参考，对每个后续帧点 $Q_i^t$ 计算：

$$
\bar P=\frac{\sum_i w_iP_i}{\sum_iw_i},\qquad
\bar Q=\frac{\sum_i w_iQ_i}{\sum_iw_i},
$$

$$
H=\sum_i w_i(P_i-\bar P)(Q_i-\bar Q)^\top=U\Sigma V^\top,
$$

$$
R_t=VU^\top,\qquad t_t=\bar Q-R_t\bar P.
$$

当 `det(R_t)<0` 时修正 SVD 反射。第一次拟合后计算逐点残差，保留：

$$
e_i < \operatorname{mean}(e)+\lambda_{outlier}\operatorname{std}(e),
$$

再对 inliers 做一次等权 SVD。可见点少于 3 个时直接沿用前一帧变换。输出 $T_{0\rightarrow t}$，即“首帧 manipulated 点云到当前帧”的刚体变换，而不是机械臂末端位姿。

### 3.4 统一重采样为 64 步

先计算相邻帧的平移与旋转增量：

$$
\Delta s_t=\|t_{t+1}-t_t\|_2+\lambda_R\,2\arccos(|q_{t+1}^{\top}q_t|),
$$

累积为轨迹参数 $S_t$，再在 $[0,S_{end}]$ 上均匀取 64 个位置。平移用 cubic/linear interpolation，旋转用 quaternion SLERP。pouring 的 `lambda_rot=0.1`；drawer 被视为平移关节，`lambda_rot=0.0`。

保存文件为：

```text
episode_x/
  point_tracking/tapip3d_result.npz
  se3_trajectory/se3_relative_trajectory.npz
  se3_trajectory/dp_action_trajectory.npz
```

其中 `actions_8d=[tx,ty,tz,qx,qy,qz,qw,gripper]`；当前 GoalPose 与 Full64 训练都只读取前 7 维，人工设置的第 8 维 gripper 状态不参与这两个模型。

## 4. 公共坐标与位姿表示

令首帧 manipulated 点云质心为：

$$c_0=\frac{1}{N}\sum_i p_i^m.$$

两个点集统一局部化：

$$P_m^{local}=P_m-c_0,\qquad P_r^{local}=P_r-c_0.$$

原始标签 $T=[R,t]$ 绕同一个质心改写为局部变换：

$$t^{local}=Rc_0+t-c_0,\qquad R^{local}=R.$$

因此对局部点 $p-c_0$ 应用 $[R,t^{local}]$，与先在相机系应用 $[R,t]$ 再减去 $c_0$ 等价。

代码同时使用两种 pose：

- `pose7d=[tx,ty,tz,qx,qy,qz,qw]`；
- `pose9d=[tx,ty,tz,r_1,r_2]`，其中 $r_1,r_2$ 是旋转矩阵前两列，按列展开为连续 6D rotation，第三列由正交化恢复。

## 5. 第一级：GoalPose 终态生成

### 5.1 数据样本

实现位于外部训练仓库：

```text
diffusion_policy_3d/dataset/goal_pose_dataset.py
diffusion_policy_3d/dataset/custom_multitask_dataset.py
```

每个 episode 只产生一个 GoalPose 样本：

```text
obs.pc_manipulated : [256, 3]
obs.pc_target      : [256, 3]
obs.agent_pos      : [7] = identity
obs.lang_token_embs: [1, 1024]
goal_pose9d        : [9] = local trajectory 最后一帧
```

这里有一个必须准确记录的实现细节：`MultiTaskSE3Dataset` 内部将 `self.num_pts` 硬编码为 64；`GoalPoseSE3Dataset` 随后把这 64 点 cyclic padding 到配置要求的 256 点。因此训练时的 `[256,3]` 实际是 64 个几何点重复 4 次，并不是 256 个独立采样点。

GoalPose normalizer 在训练 episode 的 local `goal_pose9d` 上以 `limits` 模式拟合，将各维映射到近似 `[-1,1]`。normalizer 随 checkpoint 保存。

### 5.2 GoalPose 网络

实现：

```text
diffusion_policy_3d/policy/goal_pose_diffuser.py
diffusion_policy_3d/model/goal/relational_pose_encoder.py
```

给定 noisy normalized pose $x_t\in\mathbb R^9$：

1. normalizer 反归一化 $x_t$，转为 $T_t$；
2. 用 $T_t$ 变换 manipulated 点云，得到 noisy candidate cloud；
3. 分别构造 static branch 和 candidate branch：
   - static：原 manipulated cloud 对 reference cloud；
   - candidate：noisy candidate cloud 对 reference cloud；
4. 每个 branch 分别减去两组点自己的质心，并除以 reference bbox 对角线长度；
5. raw xyz 经 `3 -> 128 -> 128` point MLP 和 role embedding；
6. manipulated token 作为 query、reference token 作为 key/value，做一次 4-head cross-attention；
7. relation tokens 经 residual FFN，随后拼接 max-pool、mean-pool 与归一化质心差，得到 $128+128+3=259$ 维 relation feature；
8. 拼接 static relation、candidate relation、128 维 timestep embedding、128 维 noisy-pose embedding 和 128 维 language projection；
9. fusion MLP 输出 256 维，再由三层 MLP 输出预测的 normalized clean pose $\hat x_0\in\mathbb R^9$。

static 与 candidate 使用两个独立的 `RelationBranch`，权重不共享。

### 5.3 GoalPose 扩散训练

当前 scheduler 是 DDIMScheduler，但训练加噪仍使用标准前向扩散：

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).$$

随机 $t\in[0,99]$，网络直接预测 clean sample，而不是噪声：

$$\hat x_0=f_\theta(x_t,t,P_m,P_r,l).$$

总损失为：

$$
\mathcal L_{goal}=
1.0\mathcal L_{9D}
+1.0\mathcal L_{trans}
+0.5\mathcal L_{SO(3)}
+0.1\mathcal L_{cloud},
$$

其中：

- $\mathcal L_{9D}=\operatorname{MSE}(\hat x_0,x_0)$，在 normalized 9D 空间计算；
- $\mathcal L_{trans}$ 是反归一化平移的 SmoothL1；
- $\mathcal L_{SO(3)}$ 是旋转测地角；
- $\mathcal L_{cloud}$ 比较预测与 GT 变换后的 manipulated points。

验证记录采样后的终态平移误差、旋转测地误差和 transformed-cloud MSE，best checkpoint 以 `val_sample_goal_pos_err_cm` 选择。

### 5.4 GoalPose 推理

从 $x_T\sim\mathcal N(0,I)$ 开始，执行 10 个 DDIM reverse steps，每步预测 clean 9D sample，最终反归一化为：

```text
goal_pose9d [B,9]
goal_pose7d [B,7]
T_goal     [B,4,4]
```

## 6. 第二级：Goal-conditioned Full64 中间轨迹生成

### 6.1 Full64 数据样本

实现位于：

```text
diffusion_policy_3d/dataset/full_trajectory_goal_dataset.py
```

Full64 直接从首帧采样像素与深度重建 64 个 manipulated 点和 64 个 reference 点。局部化后，将 64 步绝对局部轨迹记为 $T_i$，定义：

$$T_i^{action}=T_{start}^{-1}T_i.$$

所以 action 不是相邻步增量，而是每一步相对 episode 起点的 pose。代码强制：

$$T_0^{action}=I,\qquad T_{63}^{action}=T_{start}^{-1}T_{goal}.$$

每条样本为：

```text
obs.agent_pos          : [1, 7]    absolute local start pose
obs.pc_manipulated     : [1,64,3]
obs.pc_target          : [1,64,3]
obs.goal_pose9d        : [1, 9]    absolute local goal
obs.goal_delta_pose9d  : [1, 9]    inv(start) @ goal
obs.goal_delta_pose7d  : [1, 7]    用于终点 hard inpainting
obs.lang_token_embs    : [1,1024]
action                 : [64,7]    每步相对 start 的 pose
```

训练增强对两组点施加同一个随机 SE(3)（Euler 各轴 ±5°，平移各轴 ±2 cm），并对轨迹做共轭变换 $T_{aug}T_iT_{aug}^{-1}$。

平移由 `limits` normalizer 处理；四元数不做数据统计，只做单位归一化。goal 9D 与 goal-delta 9D 另有各自 normalizer。

### 6.2 当前真正使用的场景编码器

虽然 `simple_dp3.py` 仍保留旧的 `ManipulationCentricSE3Encoder` 和基础 DP3 encoder 分支，pouring 与 drawer 当前 checkpoint 的 resolved config 都是：

```yaml
obs_encoder_type: goal_conditioned_set_transformer
use_cross_attention: false
```

所以当前生效的是 `GoalConditionedSetTransformerEncoder`：

1. 64 个 manipulated points 与 64 个 reference points 分别经过 6-band 3D Fourier positional encoding；
2. 共享 point MLP 投影为 128 维，并添加 object-role embedding；
3. 每组点各用 16 个 learned queries 做 attention pooling，得到 `16 + 16` 个 scene tokens；
4. 另外构造 CLS、start pose、goal delta 9D、goal absolute 9D 和 language token；
5. 总计 37 个 token 输入 3 层 Transformer Encoder（hidden 128、4 heads、FFN 512、dropout 0.1）；
6. 只取 CLS，经 MLP 输出一个 256 维 `global_cond`。

这 256 维向量是后续 64 个时间位置共享的唯一场景/目标条件。

### 6.3 64 步时序去噪网络

`ConditionalUnet1D` 的输入状态为：

$$X_t\in\mathbb R^{B\times64\times7}.$$

网络沿 64 步时间轴做 1D convolutional U-Net：

- diffusion timestep embedding：256；
- down channels：128、256、384；
- kernel size：5；
- GroupNorm groups：8；
- 256 维场景条件通过 FiLM 注入 down、middle 和 up blocks。

起点与终点在所有去噪步中被强制写入：

```text
X[:, 0]  = normalized identity pose
X[:, 63] = normalized goal_delta_pose7d
```

### 6.4 Full64 训练目标

训练时对整条 normalized 7D trajectory 加高斯噪声，然后覆盖首尾边界。当前同样使用 `prediction_type=sample`，所以网络预测 clean trajectory $X_0$。损失是所有非边界元素的等权 MSE：

$$
\mathcal L_{traj}=
\frac{\sum (1-M)\odot\|\hat X_0-X_0\|^2}
{\sum(1-M)},
$$

其中 $M$ 是首尾 condition mask。训练损失只记录一个 `bc_loss`，没有独立的 translation、rotation、smoothness、collision 或 relation loss。

验证额外计算中间 62 帧与完整 64 帧的平移/旋转误差、P50/P90/max，以及第 16/32/48 步误差。best checkpoint 以 `val_sample_middle_traj_pos_err_cm_mean` 选择；硬写入的第 0/63 帧不作为主选择指标。

### 6.5 Full64 推理

从 $X\sim\mathcal N(0,I)$ 开始做 10 个 DDIM steps，每步：

1. 写入首尾边界；
2. 用 1D U-Net 预测 clean trajectory；
3. scheduler 更新；
4. 再写入首尾边界。

最终反归一化平移、归一化四元数，得到 `[B,64,7]`。

## 7. LFV 中两级模型如何串联

主入口是 [infer_functional_motion.py](../../scripts/inference/infer_functional_motion.py)，几何接口位于 [two_stage_pouring.py](../../lfv/inference/functional_motion/two_stage_pouring.py)。

### 7.1 仿真输入点云

manipulated 输入不是直接对完整 object mask 均匀采样，而是：

1. 在 object mask、有有效 depth 且 heat ≥ 0.05 的像素中取 heat 的 60% quantile；
2. 仅保留高于阈值的 contact-hot pixels；
3. 使用“heat score × 图像空间最远点距离”的确定性 greedy sampling 取 256 点。

reference 则在 reference mask 有效深度像素中做图像空间分布采样 256 点。

GoalPose 使用 256/256 点；Full64 直接使用这两个有序集合的前 64 点，并重新计算 manipulated centroid。代码显式把 GoalPose 输出从 256 点质心坐标约定转换到 64 点质心约定。

### 7.2 Goal 到 Full64

仿真推理将起点设置为 identity，因此同一个预测 goal 同时填入：

```text
goal_pose9d
goal_delta_pose9d
goal_delta_pose7d
```

Full64 生成 64 个 local relative poses 后，将每个 pose 先恢复为 camera-frame rigid delta：

$$t^{cam}=t^{local}+c-Rc,$$

再用相机外参共轭到 world frame：

$$T_\Delta^{world}=T_{cam\rightarrow world}T_\Delta^{cam}T_{world\rightarrow cam},$$

$$T_{object,t}^{world}=T_\Delta^{world}T_{object,0}^{world}.$$

### 7.3 object trajectory 到 TCP trajectory

抓取完成时计算并固定：

$$T_{object\rightarrow tcp}=T_{object,0}^{-1}T_{tcp,grasp}.$$

随后每一步：

$$T_{tcp,t}=T_{object,t}T_{object\rightarrow tcp}.$$

[execute_functional_motion_maniskill.py](../../scripts/robot/execute_functional_motion_maniskill.py) 依次执行整条轨迹。当前没有在每个 Full64 step 重新观测物体并重新规划。

drawer 执行还有一个可选的任务规则：把预测轨迹投影到已知 prismatic axis，强制位移非负、单调、可选最大拉出距离，并把所有旋转固定为初始旋转。这是执行后处理，不是 Full64 网络输出本身。

## 8. 当前代码索引

| 功能 | 文件 |
|---|---|
| mask 内首帧点采样 | [lfv/pipeline/sample_points.py](../../lfv/pipeline/sample_points.py) |
| TAPIP3D 3D tracking | [lfv/pipeline/tracking.py](../../lfv/pipeline/tracking.py) |
| weighted SVD 与 64 步标签 | [lfv/pipeline/se3_trajectory.py](../../lfv/pipeline/se3_trajectory.py) |
| 两阶段推理入口 | [scripts/inference/infer_functional_motion.py](../../scripts/inference/infer_functional_motion.py) |
| 坐标与采样适配 | [lfv/inference/functional_motion/two_stage_pouring.py](../../lfv/inference/functional_motion/two_stage_pouring.py) |
| TCP 绑定与 drawer axis 投影 | [lfv/robot/panda_grasp_execution.py](../../lfv/robot/panda_grasp_execution.py) |
| ManiSkill 执行 | [scripts/robot/execute_functional_motion_maniskill.py](../../scripts/robot/execute_functional_motion_maniskill.py) |
| GoalPose dataset/model | `/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/{dataset/goal_pose_dataset.py,policy/goal_pose_diffuser.py}` |
| Full64 dataset/model | `/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/{dataset/full_trajectory_goal_dataset.py,policy/simple_dp3.py}` |
| Goal relation encoder | `/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/model/goal/relational_pose_encoder.py` |
| Full64 scene encoder | `/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/model/vision/cross_attention_encoder.py` |
| Full64 temporal U-Net | `/home/users1/ljian/object_centric_diffusion/diffusion_policy_3d/model/diffusion/simple_conditional_unet1d.py` |

## 9. 容易混淆但需要固定的事实

- Stage 2 网络不读取逐点 DINOv2 feature。Grounding DINO/SAM2/TAPIP3D 是离线 mask/标签构造工具；网络条件是 xyz 点、终态 token 和任务级语言 embedding。
- 当前 Full64 checkpoint 使用 Set Transformer，不使用代码中保留的手工 KNN `ManipulationCentricSE3Encoder`。
- 生成的是 manipulated object/part 的运动，不是机器人关节轨迹，也不是相邻步 velocity。
- GoalPose 与 Full64 都是 clean-sample prediction，而不是 epsilon prediction。
- 当前执行是一次性 GoalPose + 一次性 Full64 的开环 rollout；它没有保留原始 SPOT 方法的在线物体 tracking 与 receding-horizon 重规划。
- drawer 的单调直线运动主要由执行阶段的 prismatic projection 显式保证，而不是由通用 SE(3) trajectory model 保证。
