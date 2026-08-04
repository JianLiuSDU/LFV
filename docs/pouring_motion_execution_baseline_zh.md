# Pouring 两阶段模型推理、Panda 执行与录像基线

更新日期：2026-08-01

## 1. 本次选择与结论

仓库外的历史训练结果中同时存在 `pickNplace` 和 `pouring` 两套两阶段模型。
检查训练数据后确认 `pickNplace_lfv` 的代表任务是香蕉到盘子，而当前场景是带把手
杯子到碗，因此本次使用用户允许的 `pouring` 模型，避免把“名称像 pick&place”
误当成“训练过杯子抓放”。所用权重为：

- GoalPoseDiffuser：epoch 700，验证采样终点位置误差 `3.086 cm`；
- goal-conditioned Full64 diffuser：epoch 1500；
- 两者都读取 pouring 的 1024 维语言嵌入并使用 EMA 权重。

本次已完成真实模型推理、Panda 控制执行和固定 MP4 录像。第一条未经任务适配的
基线失败：夹爪到达了 GraspNet 位姿但未形成双指有效抓取，因此杯子没有随预测
轨迹移动；该失败录像被保留作为后续修改的对照，而没有通过人工移动杯子伪造成功。

## 2. 固定场景

使用 `Cole_Hardware_Mug_Classic_Blue` 扫描杯和 ManiSkill YCB bowl：

```text
cup_xy  = [0.04, -0.12]
bowl_xy = [0.06,  0.18]
planar distance = 0.3007 m
cup_yaw = -90 deg
```

此前场景间距约 `0.20 m`，本次增大到约 `0.30 m`。杯把仍朝向相机画面的左侧，
相机、杯资产、随机种子和抓取 object-frame 坐标保持一致，因此可以单独观察目标
距离变化对运动模型的影响。

快照新增并固定保存：`cup_mask`、`bowl_mask`、二者采样像素/相机点、深度、内参、
`T_world_to_camera`、杯子完整表面及初始 `T_object_to_world`。

## 3. 完整计算流程

```text
ManiSkill far scene RGB-D
        │
        ├─ cup RGB/mask ── Soft Heatmap AffCorrs ── contact heat
        │                                             │
        │                      valid depth + heat-FPS ─┘
        │                              256 contact points
        │
        └─ bowl mask + valid depth ── spatial FPS ── 256 target points
                                      │
                                      ▼
        GoalPoseDiffuser: [256,3] + [256,3] -> local goal SE(3)
                                      │
                 centroid convention conversion (256 -> 64)
                                      │
                                      ▼
        Full64 Diffuser: [64,3] + [64,3] + goal -> 64 local SE(3)
                                      │
                camera-local delta -> camera delta -> world object poses
                                      │
GraspNet object-frame grasp ── frame conversion ── Panda TCP grasp
                                      │
       T_tcp(i) = T_object(i) inv(T_object(0)) T_tcp(grasp)
                                      │
                                      ▼
       pregrasp -> top-down approach -> close -> learned 64-step path
                                      │
                           MP4 + keyframes + JSON metrics
```

### 3.1 模型输入 shape 与归一化坐标

第一阶段 checkpoint 的真实训练 shape 是：

```text
pc_manipulated [B,256,3]
pc_target      [B,256,3]
agent_pos      [B,7]
lang           [B,1,1024]
```

第二阶段 checkpoint 的真实训练 shape 是：

```text
pc_manipulated      [B,1,64,3]
pc_target           [B,1,64,3]
agent_pos            [B,1,7]
goal_pose9d          [B,1,9]
goal_delta_pose9d    [B,1,9]
goal_delta_pose7d    [B,1,7]
lang                 [B,1,1024]
output action_pred   [B,64,7]
```

两组点云都使用同一个 manipulated centroid：

```text
pc_man_local = pc_man_camera - C
pc_tgt_local = pc_tgt_camera - C
```

第一阶段使用 256 点 centroid，第二阶段使用 64 点 centroid。终点位姿不能直接
照抄，而要先还原为相机刚体增量，再围绕第二阶段 centroid 重新参数化。历史推理
脚本曾把第一阶段强制成 64 点；本入口按 checkpoint 的 256 点真实契约修正。

### 3.2 坐标变换

模型位姿四元数顺序是 `xyzw`；ManiSkill actor 原始 pose 是 `wxyz`。模型输出的
centroid-local 位姿 `[R,t_local]` 转相机增量：

```text
t_camera = t_local + C - R C
D_camera = [R, t_camera]
D_world  = inv(T_world_to_camera) D_camera T_world_to_camera
T_object_world(i) = D_world(i) T_object_world(0)
```

GraspNet rotation 列定义为 `[approach, closing, vertical]`，Panda TCP rotation 列
定义为 `[cross(closing,approach), closing, approach]`。GraspNet translation 位于
接触中心后方一个 decoded depth，Panda TCP 则锚定到接触中心。闭合以后保持：

```text
T_object_to_tcp = inv(T_object_world(0)) T_tcp_grasp_world
T_tcp_world(i)  = T_object_world(i) T_object_to_tcp
```

Panda 使用 ManiSkill `pd_ee_pose` 组合控制器。这里存在一个容易忽略的混合契约：
机械臂六维动作是未归一化的 base-frame 绝对 XYZ + Euler XYZ，而最后一维夹爪
动作仍被归一化，`+1.0` 是完全张开，`-1.0` 才是控制器下限/完全闭合。实际双指
关节位置需另外从 robot qpos 的最后两维读取。

## 4. 第一条基线结果

二维迁移被接受，global confidence 为 `0.4301`，热力位于杯把。模型预测最终杯子
世界位置约 `[0.0623, 0.0237, 0.1693] m`，相对旋转 `114.38°`；虽然表现出抬升
和倾倒意图，但目标 bowl 的 y 为 `0.18 m`，预测终点在 y 方向仍相差约
`0.156 m`。

执行指标：

```text
initial cup-bowl planar distance     0.3007 m
snapshot/execution alignment error  < 0.000001 m
mean TCP position tracking error     0.00586 m
max TCP position tracking error      0.01894 m
grasp acquired after close           false
grasped at end                       false
simulator success                    false
video                                1280x720, 30 FPS, 369 frames, 12.3 s
```

关键诊断是 `grasp acquired after close=false`。静态 GraspNet 检查验证的是采样点云
中的指尖热力、top-down 角度和几何碰撞 IoU；它并不保证 Panda 指垫在 PhysX convex
collision mesh 上产生满足 `is_grasping` 的双侧接触。录像关键帧确认夹爪到达杯把
附近但没有把手可靠夹在两指之间。此问题优先于运动模型优化；若夹取未成立，后续
轨迹质量无法通过物体运动评价。第二个独立问题是模型终点没有到达远处 bowl，说明
新扫描杯、ManiSkill 相机/几何及 30 cm 目标距离对历史 MuJoCo pouring 数据构成了
明显域外输入。

### 4.1 完全闭合修正与双视角重跑

检查第一条录像后发现，旧执行器错误地把 `-0.01` 当作夹爪物理关节目标；但在
Panda `pd_ee_pose` 组合动作中它实际是归一化值，因此只给出了接近中位的命令。
修正后的执行顺序为：

```text
fully open (+1.0)
  -> pregrasp
  -> top-down grasp pose
  -> 45 steps full close (-1.0)
  -> 20 steps hold full close (-1.0)
  -> Full64 trajectory, every step continues sending -1.0
```

重跑读取完全相同的 snapshot、二维热力和模型预测，没有重新采样或改变运动轨迹。
闭合前实际双指 qpos 为 `[0.0399998, 0.0399998] m`，闭合保持后为 `[0,0] m`，
证明“完全闭合”已真实执行，而不只是修改画面文字。重跑仍然
`grasp_acquired_after_close=false`：正前方关键帧显示夹爪位姿偏在把手外侧，完全
闭合时推动了杯子但没有让把手处在双指之间。因此当前剩余问题已从“闭合命令错误”
收敛为“GraspNet frame 到 Panda TCP 的接触中心/横向偏置不正确”。

本次每个仿真步同步保存两条录像：

- `pouring_model_execution.mp4`：1280×720 斜前方 render camera；
- `pouring_model_execution_front.mp4`：640×480 正前方 `base_camera`，也是推理输入视角；
- 两者均为 30 FPS、410 帧、13.67 秒，并各自保存 9 张同帧关键图。

### 4.2 TCP 偏移标定与快速效果重跑

为把静态 GraspNet 接触中心可靠转换为 Panda 的实际指垫夹持位置，新增了只使用
状态观测的 7×7 局部偏移搜索。每个候选都独立 reset，并执行“张开—预抓取—
接近—完全闭合—垂直抬升 6 cm”；只有闭合及抬升后的 `is_grasped`、指间余量和
杯子实际高度变化参与选择。偏移坐标固定为 Panda TCP 局部
`[orthogonal, closing, approach]`。49 个候选中唯一通过抬升验证的是：

```text
grasp_offset_local = [+0.005, -0.005, 0.000] m
finger qpos after close = [0.010634, 0.010593] m
grasped after close/lift = true / true
```

该偏移已写入实验 YAML，并用原 snapshot、原 GraspNet 候选和原 Full64 预测重新
录像。快速重跑在闭合后得到 `grasp_acquired_after_close=true`，实际双指 qpos 为
`[0.010558, 0.010517] m`，说明杯把确实处于两指之间，而不是空闭合；但旧 Full64
轨迹开始后在视频第 135 帧丢抓，最终任务仍失败。因此抓取入口的两个执行错误
（夹爪命令尺度、TCP 横向偏移）已经修正，当前瓶颈已经转移到历史运动模型的早期
轨迹动力学/域外泛化，而不是接触热力或闭合指令。

快速效果输出为 338 帧、30 FPS、11.27 秒：

- `pouring_model_execution_front.mp4`：640×480 正前方主检查视角；
- `pouring_model_execution.mp4`：1280×720 斜前方辅助视角；
- `keyframe_front_grasp_closed.png`：固定记录闭合后 `is_grasping=true` 的证据帧；
- `execution_report.json`：记录偏移、实际指关节、首次丢抓帧及双路产物路径。

### 4.3 UMI/Fin-Ray 风格长指改造

进一步检查表明，短指虽然能在静态闭合时夹住杯把，但原 Panda 指垫有效接触面仅
约 `17.5×18.5 mm`，在 Full64 的抬升和大角度旋转初段容易滑脱。参考资料需要
区分清楚：BridgeACT 的实机配置是 **Franka + UMI gripper**；本地
im2Flow2Act 仿真则是 **UR5e + WSG50 + wide Fin-Ray finger**，其 MJCF 同时使用
约 16 cm 的宽型长指视觉 mesh、专用 collider 和摩擦系数 5。二者不是同一个法兰
或控制器，但共同的机械原则是“平行夹持 + 长接触面 + 柔顺/高摩擦表皮”。

LFV 第一版没有直接替换整套 WSG50，因为这会改变 Panda 法兰、关节、开口、TCP
和全部现有抓取位姿；而是在两个原 Panda 移动指 link 上加入刚性长指套：

```text
robot uid                 panda_long_finger
each contact plate        30 × 70 × 8 mm
contact area              2100 mm² (stock 323.75 mm², 6.49×)
static/dynamic friction   5.0 / 5.0
TCP and opening           unchanged / 80 mm
plate centre              stock panda_hand_tcp contact height
```

视觉 box、碰撞 box 和高摩擦 PhysX material 被加入同一个 articulation finger
link，因此不是只改变录像外观；`Panda.is_grasping` 读取的左右接触力真实包含长指
碰撞。黑色主体和橙色内侧条用于在视频中辨认接触面。它是 TPU/Fin-Ray 柔顺指的
第一版刚性近似，并不声称模拟了软体变形。

相同 GraspNet 位姿、相同 `[+5,-5,0] mm` TCP 偏移的闭合—抬升探针得到：

```text
grasped after close / lift       true / true
finger qpos after close          [0.011668, 0.011626] m
cup lift in 6 cm probe           0.04543 m
short-finger calibrated probe    0.02324 m
```

随后复用完全相同的 snapshot、热力、抓取候选和 Full64 轨迹重跑。长指版本在全部
64 个模型 waypoint 以及 final hold 后始终保持 `is_grasping=true`，不再出现原来
第 135 帧丢抓；实际最终杯子高度为 `0.1301 m`。`simulator_success` 仍为 false，
是因为历史模型预测终点没有真正到达 30 cm 外的 bowl，而非再次滑脱。这把当前
问题明确拆成了“抓取保持已解决”和“运动目标预测仍需改进”两部分。

参考实现：

- BridgeACT：<https://arxiv.org/abs/2604.23249>
- im2Flow2Act：<https://github.com/mengdaxu/im2Flow2Act>
- UMI gripper：<https://umi-gripper.github.io/>

## 5. 固定入口与产物

一键复现：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/run_pouring_motion_execution.py \
  --config configs/experiments/functional_motion/pouring_cup_far_execution.yaml
```

可用 `--skip-snapshot`、`--skip-transfer`、`--skip-motion`、`--skip-execution`
复用上游保存结果，便于只改某一层。输出根目录：

```text
/home/users1/ljian/lfv_runs/pouring_motion_execution/cole_blue_mug_far_seed_0/
├── snapshot/          RGB-D、cup/bowl mask、点云和坐标系
├── transfer/          连续热力、置信度和六联图
├── motion_inference/  goal/Full64 输出、世界物体轨迹和叠加图
├── execution/         原始错误夹爪命令基线，保留用于对比
├── execution_full_close_front/
│   ├── pouring_model_execution.mp4
│   ├── pouring_model_execution_front.mp4
│   ├── keyframe_*.png
│   ├── keyframe_front_*.png
│   └── execution_report.json
├── grasp_offset_calibration.json       49 次状态标定及最终偏移
├── execution_calibrated_grasp_short_close_front/
│   ├── pouring_model_execution.mp4
│   ├── pouring_model_execution_front.mp4
│   ├── keyframe_*.png
│   ├── keyframe_front_*.png
│   └── execution_report.json
├── grasp_offset_long_finger_probe.json
├── execution_long_finger_front/
│   ├── pouring_model_execution.mp4
│   ├── pouring_model_execution_front.mp4
│   ├── keyframe_*.png
│   ├── keyframe_front_*.png
│   └── execution_report.json
└── logs/              四个阶段的完整 stdout/stderr
```

后续改进必须使用新输出目录或保留本目录，不能覆盖这些诊断证据。下一步应在
长指稳定抓取的前提下，检查/重采样 GoalPose/Full64 的目标位置、速度、姿态跳变和
碰撞；之后做多 seed 目标位姿采样和可达/目标距离筛选，最后比较运动模型重训或
域适配。若走向真实 Franka，必须把当前刚性高摩擦近似替换为可制造的 TPU/硅胶
长指，校核附加质量、挠曲、力矩、桌面间隙，并重新标定 TCP 和 grasp force。
