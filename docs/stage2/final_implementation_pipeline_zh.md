# Stage 2 V2：最终实现、训练、仿真推理与执行流程

> 实施日期：2026-08-07
> 模型注册名：`three_token_hierarchical_diffusion`
> 任务：pouring（蓝色 Cole mug → bowl）
> 本文描述 LFV 仓库内当前实际运行的实现，不再描述历史
> `object_centric_diffusion` checkpoint。

## 1. 本阶段完成的闭环

Stage 2 现在是 LFV 内可独立训练、采样、测试、仿真推理和执行的两级生成系统：

```text
初始 RGB-D + manipulated/reference mask
  -> 两组各 256 个同索引 XYZ + DINOv2 descriptor
  -> 双对象 Scene Encoder
  -> 3 个 scene context tokens
  -> Goal Pose Diffusion：K_g 个 9D 终态
  -> Trajectory Diffusion：每个 goal 生成 K_t 条 64×9D 轨迹
  -> 无 GT 的候选排序
  -> camera-local delta -> world object poses
  -> 64 帧坐标系叠加图
  -> Stage 1 已生成的 object-frame GraspNet 抓取
  -> 固定 object-to-TCP attachment
  -> Panda long-finger 抓取、完全闭合、执行轨迹并录制双视角视频
```

系统边界保持明确：网络学习的是 **manipulated object 的 object-centric
功能运动先验**；GraspNet、夹爪闭合、IK/control、碰撞与执行监控仍属于后级。
Contact heat 不作为 Stage 2 输入，Stage 1 与 Stage 2 只共享 DINO 语义基础与
像素—点坐标约定。

## 2. 数据审查与修复结果

源数据保持只读：

```text
/media/ljian/lj/data_3d/pouring_lfv
```

派生缓存位于：

```text
/home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1
```

审查 180 个 episode，179 个通过；`episode_7` 缺少
`se3_trajectory/dp_action_trajectory.npz`，因此拒绝。修复了历史链路中“64 个
点循环补成 256 个点”的问题，也没有直接复用包含无效深度的旧 2D 采样。对每个
对象执行以下固定流程：

1. 在首帧 mask 内同时过滤有限深度与 0.1–2.0 m 工作范围；
2. 用确定性 image-space FPS 选择正好 256 个互不重复像素；
3. 同一像素索引同时用于深度反投影 XYZ 和 DINO 网格双线性采样；
4. DINO descriptor 逐点 L2 归一化，并以 float16 离线保存；
5. 训练时 points 与 DINO 只允许联合置换，两个对象可独立置换；
6. Goal 和 Trajectory 共用同一批点、同一个 manipulated centroid 与 scale。

最终 179 条缓存全部满足：

```text
manipulated_points       [256,3]   float32
manipulated_dino         [256,384] float16 (读取后 float32)
reference_points         [256,3]   float32
reference_dino           [256,384] float16 (读取后 float32)
goal_pose9d              [9]       float32
trajectory_pose9d        [64,9]    float32
scene_origin             [3]
scene_scale              scalar
episode_id / object_instance_id / source_fingerprint
```

当前源数据没有真实 `object_instance_id` 元数据，所以 split manifest 明确标记为
`episode_split_baseline`：143 train / 18 val / 18 test。它能防 episode 泄漏，
但不能把当前结果宣称为严格的 instance-disjoint 泛化；补齐实例 ID 后，已有 split
builder 会按实例整体划分。

## 3. 坐标系、Pose9D 与归一化

两组点均处于 OpenCV camera frame，使用 manipulated 256 点质心 $c$：

$$
p_{local}=\frac{p_{camera}-c}{s},\qquad s=1.0.
$$

原始轨迹标签是 camera frame 中左乘物体点的相对变换
$T_c=[R,t_c]$。变到共享 local frame：

$$
t_{local}=\frac{Rc+t_c-c}{s}.
$$

推理时严格做逆变换：

$$
t_c=s\,t_{local}-Rc+c.
$$

每个位姿使用 9D：

```text
[translation_xyz, rotation_column_0, rotation_column_1]
```

旋转 6D 是旋转矩阵的前两列，解码时用 Gram–Schmidt 恢复 $SO(3)$；不对旋转
做均值方差标准化。Normalizer 只使用 train split 的全部轨迹帧统计 translation
三个维度的 mean/std，并随 checkpoint 精确保存。EMA 只平均可学习参数，绝不
平均 Normalizer 统计量。

## 4. Scene Encoder：两个对象、三个上下文 Token

输入：

```text
manipulated_points [B,256,3]    manipulated_dino [B,256,384]
reference_points   [B,256,3]    reference_dino   [B,256,384]
```

### 4.1 逐点输入与两个 PointNet 分支

- 共享 DINO projector：`LayerNorm(384) -> Linear(384,256) -> GELU -> Linear(256,64)`；
- manipulated/reference 各自独立 XYZ projector：`Linear(3,64) -> GELU`；
- 每点拼接为 128 维；
- 两个不共享权重的 PointNet branch：三层逐点 MLP，输出
  $F_m,F_r\in\mathbb R^{B\times256\times128}$；
- 对点维 max pooling 得到 $g_m,g_r\in\mathbb R^{B\times128}$。

不共享 PointNet 是为了让两个角色学习不同关注方式；共享 DINO projector 则保证
两侧语义 descriptor 落在同一个投影空间。

### 4.2 三个上下文 Token

1. 初始布局 token：

   $$z_{init}=MLP([g_m,g_r]).$$

2. manipulated 查询 reference：

   $$\tilde F_m=MHA(Q=F_m,K=F_r,V=F_r),$$

   将 $[F_m,\tilde F_m]$ 经 fusion MLP 后 max pool，得到 $z_{m\leftarrow r}$。

3. reference 查询 manipulated：使用另一套独立 cross-attention，得到
   $z_{r\leftarrow m}$。

加上三个可学习 type embedding 后：

```text
context_tokens = [z_init, z_m<-r, z_r<-m]  # [B,3,128]
```

debug 模式保留两张 `[B,4,256,256]` attention map，并把 key 维平均后形成
manipulated/reference importance，可叠加回仿真截图；这些 debug map 不参与
后续扩散输入。

## 5. Goal Pose Diffusion

### 5.1 状态与加噪

干净状态是一个 9D 终态 $x_0^G\in\mathbb R^{B\times9}$。translation 先经
Normalizer，rotation6D 保持原值；每个 batch 随机采样一个 DDPM timestep：

$$
x_t^G=\sqrt{\bar\alpha_t}x_0^G+\sqrt{1-\bar\alpha_t}\epsilon.
$$

Scheduler 使用 cosine beta、100 个训练步、`prediction_type=sample`；也就是网络
直接预测干净 $\hat x_0^G$，而不是预测噪声。

### 5.2 Goal Transformer block

`GoalPoseDecoder` 先执行 `9 -> 128 -> 128` pose embedding。扩散 timestep 经
sinusoidal embedding 和两层 MLP 得到 128 维条件。第一版堆叠 4 个 block：

```text
noisy goal token [B,1,128]
  -> timestep-conditioned AdaLayerNorm
  -> 4-head cross-attention to scene context [B,3,128]
  -> residual
  -> timestep-conditioned AdaLayerNorm
  -> FFN(128 -> 512 -> 128)
  -> residual
```

只有一个 goal token 不妨碍扩散：多模态来自不同初始 Gaussian state；单 token
仍可在每个去噪步查询三类场景关系。9D 是刚体终态的最小连续任务状态之一，逐点
输出只在目标本身不是刚体或需要形变/contact distribution 时才必要。

### 5.3 Goal loss 与采样

$$
\mathcal L_G=
\operatorname{MSE}(\hat x_0^G,x_0^G)
+\operatorname{SmoothL1}(\hat t,t)
+0.5\,d_{SO(3)}(\hat R,R).
$$

推理使用 20-step DDIM，从独立 Gaussian state 生成默认 16 个 goal 候选。

## 6. Goal-conditioned Trajectory Diffusion

### 6.1 状态、goal uncertainty 与起点

完整轨迹是 `[B,64,9]`。第 0 帧永远是 identity，不参与扩散；扩散状态只包含
第 1–63 帧，即 `[B,63,9]`。训练时 34% 使用干净 GT goal，66% 给 normalized
GT goal 加标准差 0.03 的小 Gaussian perturbation，使轨迹 decoder 不只适配完美
goal。推理时输入 Goal Diffusion 实际生成的多个 goal，每个 goal 再生成多条轨迹。

### 6.2 Trajectory Transformer block

每个 noisy pose 先经 `9 -> 128 -> 128` embedding，并加上固定 1/63…1 的
sinusoidal progress embedding。memory 是：

```text
[3 scene context tokens, embedded goal token]  # [B,4,128]
```

第一版堆叠 6 个非因果 block，每层为：

```text
AdaLN(t) -> temporal Conv1d(kernel=3) -> residual
AdaLN(t) -> 4-head non-causal temporal self-attention -> residual
AdaLN(t) -> cross-attention to [scene x3 + goal x1] -> residual
AdaLN(t) -> FFN(128 -> 512 -> 128) -> residual
```

这里没有照搬 Goal block：局部 Conv1d 明确学习相邻运动连续性；非因果 self-attention
让每个时间 token 同时看到整条路径；cross-attention 让每一步直接查询场景和 goal，
而不是先把所有条件压成一个向量。最终线性头预测 63 个干净 Pose9D，再在最前面
拼回 identity。

### 6.3 Trajectory loss 与采样

start-fixed v3 仍扩散 frame 1–63，但把已知 identity frame 0 作为不可修改的条件
token 放进每层 decoder。扩散重建对 frame 1 使用 20 倍权重、对 frame 63 使用
2 倍权重。当前物理损失为：

$$
\begin{aligned}
\mathcal L_T={}&\mathcal L_{diff}^{w_1=20,w_{63}=2}
+\operatorname{SmoothL1}(\hat t_{1:63},t_{1:63})
+0.5\,\overline{d_{SO(3)}(\hat R,R)}\\
&+0.5\,\operatorname{L1}(\Delta\hat t,\Delta t)
+0.1\,\operatorname{L1}(\Delta^2\hat t,\Delta^2t)\\
&+2\left(\operatorname{MSE}(\hat x_1^n,x_1^n)
+\operatorname{L1}(\hat t_1,t_1)
+0.5d_{SO(3)}(\hat R_1,R_1)\right)\\
&\quad
+\left(\operatorname{L1}(\hat t_{63},t_G)
+0.5d_{SO(3)}(\hat R_{63},R_G)\right).
\end{aligned}
$$

最后一帧仍是 soft endpoint constraint，不在推理后手工覆盖为 goal，因此轨迹可以在
goal uncertainty 下联合调整并保持连续。统一采样接口可分别覆盖 goal/trajectory
DDIM 步数；test 固定为 20 步，最终蓝杯诊断使用 50 步验证充分去噪。20/50 步的
蓝杯首段分别为 8.45/8.44 mm，说明剩余误差主要不是 DDIM 步数不足。

### 6.4 轨迹不是相邻帧残差：起点跳变问题的复核

这里必须区分两个容易混淆的“relative/residual”概念：

- 原始 `dense_se3_from_tracking` 固定 $P_{ref}=P_0$，每个时刻都重新求
  $P_0\rightarrow P_k$ 的刚体变换；
- 因此标签第 $k$ 帧是累计的相对首帧变换 $T_{0\rightarrow k}$，不是
  $T_{k-1\rightarrow k}$ 的相邻帧残差；
- `trajectory_pose9d[k]` 直接编码 $T_{0\rightarrow k}$；Goal 直接取最后一个
  $T_{0\rightarrow63}$；
- 推理恢复为
  $T_{object,w}^{(k)}=T_{\Delta w}^{(k)}T_{object,w}^{(0)}$ 是正确的。translation
  可以直观理解为相对首帧位移，但有旋转和相机/world 坐标变换时必须做完整矩阵
  复合，不能只对 xyz 做普通加法；也绝不能把 64 个 $T_{0\rightarrow k}$ 再逐帧累乘。

所以当前终态是直接监督的累计终态，不存在相邻残差积分造成的终态漂移。当前
Goal 收敛/多样性问题主要来自 scene conditioning 与生成塌缩，而不是残差标签。

不过 v2 Trajectory Decoder 存在明确的 **start-boundary discontinuity**：训练时
只扩散 frame 1–63，identity frame 0 是采样完成后才拼回；decoder 内部没有一个
参与 attention 的硬 identity token。现有 velocity loss 虽然比较了
$[0,\hat t_1,\ldots]$，但对全部 63×3 项取均值且权重仅 0.2，第一步只占约 1/63；
diffusion reconstruction 还只给最后一帧 2 倍权重，没有提高第一帧权重。

实际复核：

- 143 条 train label 的首步 translation：median 0.53 mm、max 2.73 mm；
- train label 首步 rotation：median 0.42°、max 2.20°；
- 四个训练集 top-1 推理首步 translation：14.9–32.7 mm；
- 蓝杯推理首步 translation：44.3 mm，首步 rotation：4.02°；
- 该蓝杯首步经 camera→world 转换后令 object world-Z 先下降约 29.0 mm。

因此可视化中的“先向下”不是 GT 动作结构，也不是把残差累加错了，而是模型首步
不连续。start-fixed v3 保留累计 $T_{0\rightarrow k}$ 表示，并已实现：

1. 把 identity frame 0 作为 hard-inpaint token 放进每个训练/去噪 block，而不是
   采样后才拼接；
2. 给 frame 1 diffusion reconstruction 设置20倍权重，并给 $T_0\rightarrow T_1$
   的 normalized pose、translation 和 SO(3) 设置2倍独立边界损失；
3. translation velocity 权重提高至0.5，并加入0.1倍二阶平滑；
4. 继续保留1倍 soft goal endpoint，不为修复首步而改成相邻残差扩散。

hard token 在每个 Transformer block 后都会重置为 clean normalized identity：
attention 可以读取起点，但噪声更新不能移动起点。采样 frame 0 仍由
`identity_pose9d` 精确构造，frame 1 则由首段重建、SE(3) 边界和加速度损失学习。
旧 checkpoint 缺少新字段时自动回退到 `hard_start=false` 与旧权重，可继续做 A/B。

若以后改成相邻增量 $\delta T_k=T_{k-1}^{-1}T_k$，推理必须逐帧复合，而且小误差会
累积到终态；必须再配 hard/soft goal correction。它适合强化局部运动平滑，但不比
当前累计表示天然更利于终态收敛，所以不应作为本问题的第一修复。

联合训练总损失：

$$
\mathcal L=\mathcal L_G+\mathcal L_T.
$$

## 7. 训练基础设施与通过门槛

训练器支持 YAML、固定 seed、AdamW、AMP、gradient clipping、cosine LR、EMA、
TensorBoard、loss 分项、last/best checkpoint、optimizer/scheduler/scaler/RNG 全状态
恢复。扩散验证每个 epoch 使用相同 timestep/noise probe，随后恢复训练 RNG，保证
best checkpoint 是同条件比较。

已完成门槛：

- Stage 2 单元测试：数据联合采样、6D rotation、DDPM/DDIM、encoder、forward、
  checkpoint 与固定 seed 采样复现全部通过；
- synthetic overfit：validation total 从 1.73 降至约 0.027；
- 真实 32 条 overfit：validation total 从 2.89 降至约 0.14，best 约 0.129；
- 完整训练与 test-set 生成指标记录在本文第 10 节。

最终回归结果为 **45/45 tests passed**（其中 Stage 2 为 8 个测试文件），且
`git diff --check` 通过。

## 8. 蓝杯仿真推理与候选选择

固定配置是：

```text
configs/stage2/blue_mug_pouring_execution.yaml
```

蓝杯位于 `[0.04,-0.12]`，把手朝相机左侧；碗位于 `[0.06,0.18]`。推理适配器
不消费 Stage 1 heat，而是在 cup mask 与 bowl mask 内分别按训练规则重采 256 个
有效像素，使用同一像素生成 XYZ+DINO。

默认生成 `16 goals × 2 trajectories`。部署没有 GT，因此排序只使用：

1. goal 相对 reference centroid 的残差在 train split 中的 z-score；
2. goal rotation magnitude 的 train prior；
3. 轨迹末端与其条件 goal 的 translation/SO(3) 一致性；
4. 二阶 translation difference 和异常大单步位移。

排序不读取仿真真值终态，也不调用成功判据。所有候选、分数和被选索引都写入
NPZ/JSON，后续可无缝替换为碰撞/IK-aware ranker。

局部轨迹依次转换为 camera delta、world delta 和绝对 object pose：

$$
T_{\Delta w}=T_{w\rightarrow c}^{-1}T_{\Delta c}T_{w\rightarrow c},\qquad
T_{object,w}^{(k)}=T_{\Delta w}^{(k)}T_{object,w}^{(0)}.
$$

可视化把 64 个附着在杯子上的坐标系投影回 base-camera RGB；X/Y/Z 分别为
红/绿/蓝，橙色折线为 object centroid path，每 4 帧标序号。

## 9. 抓取、执行与视频

执行复用 Stage 1 FGW + complete-surface GraspNet 生成的 object-frame top-down
抓取。该表示不依赖杯子在世界中的平移，因此可在相同资产的新 snapshot 中通过
初始 $T_{object\rightarrow world}$ 恢复 world TCP grasp。

执行顺序固定为：

```text
initial hold
-> move to pregrasp (gripper open)
-> collision-checked preshaped approach
-> reach grasp pose
-> FULL CLOSE command -1.0
-> close hold
-> object trajectory x fixed object-to-TCP attachment
-> final hold
```

使用 `panda_long_finger` 扩大把手夹取接触面积；录像同时保存 oblique render 与
base-camera front view。报告包含抓取是否获得/丢失、TCP tracking error、预测与实际
末端物体误差和 simulator success。当前仍是开环轨迹跟踪，不能把训练 loss 当成
无碰撞或稳定执行保证。

## 10. v2 基线结果与产物

### 10.1 训练

完整训练运行 244 个 epoch，并在 epoch 243 按 patience=80 正常 early stop。
固定 probe 的 validation total 从 epoch 0 的 2.006 降至 best epoch 163 的
**0.42428**。best/last checkpoint 均已保存；best checkpoint 为约 73 MB。

### 10.2 Test split 的 16×2 配对采样

18 个 test episode 的结果如下。单位分别为 m 与 degree：

| 权重 | Goal top-1 t / R | Goal best t / R | Traj top-1 mean t / R | Traj best mean t / R |
|---|---|---|---|---|
| EMA | 0.02649 / 26.72 | 0.02635 / 26.70 | 0.04073 / 14.90 | 0.03944 / 14.83 |
| raw | 0.02694 / 25.61 | 0.02681 / 25.59 | 0.04088 / 14.21 | 0.03978 / 14.16 |

EMA 平移略优，raw 旋转与综合误差略优，因此正式蓝杯 rollout 使用 raw。两份完整
JSON 位于 `full_joint_v2/test_metrics_ema.json` 和 `test_metrics_raw.json`。

### 10.3 蓝杯推理

固定 seed=42 生成 16 goals × 2 trajectories，选择 goal 7 / trajectory 0：

- goal rotation：104.15°；
- trajectory endpoint 与 goal：translation 1.91 cm、rotation 3.87°；
- 最大单步 translation：4.43 cm；
- 预测最终相对旋转：105.05°；
- 预测最终 object world position：`[-0.0347, 0.0473, 0.1429] m`。

需要保留的诊断是：16 个 goal translation 的跨样本标准差只有约
0.12–0.20 mm，Best-of-K 改善也很小，说明第一版出现明显 mode collapse；选中
goal 的 relation-residual prior score 也较高。当前输出仍有“抬升—接近—倾倒”的
正确大结构，但后续应优先改善 scene conditioning 和生成多样性，而不是把当前
结果解释成已经完成同类别多模态泛化。

产物：

```text
/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/blue_mug_seed_0/motion_inference/
├── functional_motion_prediction.npz
├── motion_inference_report.json
├── full64_coordinate_frames_overlay.png
└── encoder_cross_attention_summary.png
```

### 10.4 抓取与执行

ManiSkill rollout 已完整执行：

- snapshot/execution 初始对齐误差：$2.98\times10^{-8}$ m；
- 到位后执行完全闭合指令，`grasp_acquired_after_close=true`；
- 全部轨迹结束仍为 `grasped_at_end=true`，没有丢抓帧；
- mean/max TCP tracking error：5.18 mm / 32.89 mm；
- actual final 与 predicted final position error：3.39 cm；
- 录像：348 帧、30 FPS、11.6 s、640×480，两个 MP4 均通过 ffprobe；
- `simulator_success=false`：杯子稳定抓起并完成倾斜，但预测终态没有充分满足
  仿真 pouring 成功区域。这与上面的 relation residual / mode collapse 诊断一致。

执行产物：

```text
/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/blue_mug_seed_0/execution/
├── pouring_execution.mp4
├── pouring_execution_front.mp4
├── execution_report.json
├── executed_trajectory.npz
└── keyframe_*.png
```

因此本轮达成的是“训练—多样本采样—图像坐标系可视化—抓取—完整运动—双视角
录像”的工程闭环，以及一次成功保持抓取的真实模型 rollout；没有把失败的 pouring
判据包装成任务成功。该失败已经被固定基础设施量化，可直接作为下一版架构的对照线。

### 10.5 训练集轨迹推理可视化复核

从 train split 按终态位移分布选取 `episode_152/90/33/12` 四条数据。汇总图每行
依次显示输入 XYZ 采样、GT 累计轨迹和 top-1 预测轨迹。GT 首步与预测首步为：

| episode | GT 首步 | 预测首步 | top-1 mean t / R | endpoint t / R |
|---|---:|---:|---:|---:|
| 152 | 0.39 mm | 14.9 mm | 1.30 cm / 4.51° | 0.75 cm / 3.59° |
| 90 | 1.40 mm | 23.7 mm | 1.97 cm / 7.45° | 1.60 cm / 5.15° |
| 33 | 0.70 mm | 32.7 mm | 2.55 cm / 5.98° | 0.87 cm / 4.57° |
| 12 | 0.97 mm | 32.4 mm | 2.96 cm / 5.38° | 1.78 cm / 4.40° |

产物位于：

```text
/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_v2/train_inference_visualization/
├── train_inference_gt_vs_top1_summary.png
├── train_inference_report.json
└── episode_*_gt_vs_top1.png
```

该实验说明模型能够复现大体路径和终态，但起始边界误差显著大于中后段，支持
第 6.4 节的诊断。

## 11. 代码入口与复现实验

主要模块：

| 职责 | 文件 |
|---|---|
| 缓存与同索引采样 | `lfv/datasets/functional_motion/cache_builder.py` |
| Dataset 与联合置换 | `lfv/datasets/functional_motion/dataset.py` |
| 双向三 token encoder | `lfv/models/functional_motion_generation/encoders/bidirectional_scene_encoder.py` |
| Goal diffusion | `lfv/models/functional_motion_generation/goal/` |
| Trajectory diffusion | `lfv/models/functional_motion_generation/trajectory/` |
| 统一模型/registry | `lfv/models/functional_motion_generation/system.py`, `registry.py` |
| scheduler/normalizer/EMA | `lfv/diffusion/` |
| train/checkpoint/log | `lfv/training/functional_motion/` |
| test 与 Best-of-K | `scripts/stage2/evaluate.py`, `lfv/evaluation/functional_motion/metrics.py` |
| 仿真推理与坐标系图 | `scripts/stage2/infer_sim_snapshot.py` |
| train GT/推理对照图 | `scripts/stage2/visualize_training_inference.py` |
| 抓取+轨迹执行 | `scripts/robot/execute_functional_motion_maniskill.py` |
| 配置化完整入口 | `scripts/run_pouring_motion_execution.py` |

训练：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/stage2/train.py --config configs/stage2/pouring_lfv.yaml
```

完整蓝杯流程（已有 checkpoint 时）：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/run_pouring_motion_execution.py \
  --config configs/stage2/blue_mug_pouring_execution.yaml \
  --skip-transfer
```

快速重跑推理和执行而不重建 snapshot：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/run_pouring_motion_execution.py \
  --config configs/stage2/blue_mug_pouring_execution.yaml \
  --skip-snapshot --skip-transfer
```

所有命令都写日志到本次 output root 的 `logs/`，不会覆盖原始视频数据或 Stage 1
中间结果。

## 12. start-fixed v3：首段边界修复、重训与复核（2026-08-07）

### 12.1 修复内容

本轮没有改变数据标签语义：64 帧仍是累计的 $T_{0\rightarrow k}$，Goal 仍是
$T_{0\rightarrow63}$，世界位姿仍按矩阵左乘恢复。修改只发生在 Trajectory
Diffusion 的起点条件和损失：

1. `TrajectoryDecoder` 在63个带噪 token 前加入 normalized identity token；
2. 每个 self/cross-attention block 后把该 token 重置为 clean embedding，形成
   hard-inpaint boundary；
3. frame 1 diffusion reconstruction 权重设为20，frame 63保持2；
4. velocity 权重由0.2提高到0.5，加入0.1倍 translation acceleration；
5. 加入2倍 start-boundary loss，同时约束 normalized Pose9D、物理平移与 SO(3)；
6. evaluator 固定报告首段 GT/预测幅值及首段平移、旋转误差；
7. simulation candidate report 保存 local/world 首段、world-Z 变化和训练首段 p95；
8. `sample` 支持分别覆盖 goal/trajectory DDIM steps，旧调用与旧 checkpoint 均兼容。

### 12.2 训练门槛与正式训练

- synthetic 32：80 epochs，validation total 从2.49降至0.0789；
- real32 overfit：180 epochs，best epoch 178，validation total 0.08949；
- real32 固定四样本采样：预测首段1.21–1.48 mm、0.51–0.86°，而旧版为
  14.9–32.7 mm；
- full split：143 train / 18 val / 18 test，epoch 216按patience=80早停，
  best epoch 136，validation total 0.44082；
- checkpoint 同时保存 raw、EMA、Normalizer、optimizer、scheduler、scaler 和 RNG。

正式 checkpoint：

```text
/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_start_fixed_v3/
├── checkpoints/best.pt
├── checkpoints/last.pt
├── history.json
├── normalizer.json
├── test_metrics_raw.json
└── test_metrics_ema.json
```

### 12.3 独立 test split 的 16×2 采样

| 权重 | Goal top-1 t / R | Traj top-1 mean t / R | 预测首段幅值 | 首段 t / R error |
|---|---|---|---|---|
| raw | 3.243 cm / 24.87° | 4.553 cm / 15.03° | 3.51 mm | 3.93 mm / 0.80° |
| EMA | 3.057 cm / 26.07° | 4.494 cm / 14.72° | 2.47 mm | 3.07 mm / 0.82° |

test GT 首段均值为1.26 mm。EMA 除 Goal rotation 外整体更优，并把首段幅值进一步
压低，因此后续固定用 EMA。与 v2 相比，边界连续性显著改善，但整体轨迹误差由
v2 raw 的4.09 cm/14.21°变为4.49 cm/14.72°，没有同步改善；不能把“修好首段”
表述成整体泛化能力提升，这仍是下一轮 scene encoder/decoder 设计的重点。

### 12.4 训练集轨迹与终态目标位姿可视化

固定 `episode_152/90/33/12`，每行分别画输入、GT累计轨迹与top-1轨迹；另存一张
GT终态与 Goal Diffusion top-1 终态坐标系对照图：

| episode | GT首段 t / R | v3预测首段 t / R | Goal top-1 t / R |
|---|---|---|---|
| 152 | 0.39 mm / 0.42° | 1.34 mm / 0.39° | 1.13 cm / 1.80° |
| 90 | 1.40 mm / 0.48° | 2.50 mm / 0.47° | 1.09 cm / 1.01° |
| 33 | 0.70 mm / 0.20° | 1.86 mm / 0.58° | 1.88 cm / 0.74° |
| 12 | 0.97 mm / 0.53° | 2.06 mm / 0.36° | 1.80 cm / 0.73° |

```text
/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_start_fixed_v3/
└── train_inference_visualization_ema/
    ├── train_inference_gt_vs_top1_summary.png
    ├── train_goal_pose_gt_vs_top1_summary.png
    ├── train_inference_report.json
    └── episode_*_{gt_vs_top1,goal_gt_vs_top1}.png
```

### 12.5 蓝色杯子仿真推理

蓝杯场景重新导出 base-camera RGB-D 和两物体 mask，保持 XYZ 与 DINO 使用同一组
256像素索引；EMA 模型使用 seed=42、16 goals×2 trajectories、50步DDIM。
选择 goal 15 / trajectory 0：

- local首段8.44 mm、0.92°，world首段9.21 mm；
- world-Z 首段为 **+3.07 mm**，不再是 v2 的 **-29.0 mm 向下跳变**；
- v2 local首段44.3 mm，本轮降低约81%；
- 训练首段p95为1.49 mm，因此仿真首段仍明显偏大，属于需要保留的sim-to-real诊断；
- 最终相对旋转95.62°，最终world position为
  `[-0.06750, 0.02592, 0.15003] m`；
- 20步和50步首段几乎相同，增加采样步数不是进一步修复的有效方向。

```text
/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/blue_mug_start_fixed_seed_0/
├── snapshot/
└── motion_inference/
    ├── functional_motion_prediction.npz
    ├── motion_inference_report.json
    ├── full64_coordinate_frames_overlay.png
    ├── goal_pose_candidates_overlay.png
    └── encoder_cross_attention_summary.png
```

### 12.6 复现命令与验证

```bash
# 正式训练
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/stage2/train.py --config configs/stage2/pouring_lfv_start_fixed.yaml

# 同一仿真截图快速重跑 Goal + Trajectory 推理
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/run_pouring_motion_execution.py \
  --config configs/stage2/blue_mug_pouring_start_fixed.yaml \
  --skip-snapshot --skip-transfer --skip-execution

# 训练集轨迹和终态对照图
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/stage2/visualize_training_inference.py \
  --checkpoint /home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_start_fixed_v3/checkpoints/best.pt \
  --cache-root /home/users1/ljian/lfv_data_cache/stage2/pouring_lfv_v1 \
  --output-dir /home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/full_joint_start_fixed_v3/train_inference_visualization_ema \
  --episodes episode_152 episode_90 episode_33 episode_12 --use-ema
```

Stage 2 的8项测试全部通过；完整项目回归结果见本轮交付说明。可视化和 JSON 报告
是固定快速迭代基础设施，不依赖人工挑选 GT，且不会写入 `/media` 原始数据。
