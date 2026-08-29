# Aubo 实机执行交付包

本文档说明如何把 LFV 在 A100 规划机上得到的相机坐标系抓取位姿和 64 步对象运动计划交给另一台连接 Aubo 机械臂的电脑执行。交付包位于：

`deployment_bundle/aubo_camera_execution/`

## 1. 推荐的系统边界

完整推理分成两个端：

```text
RGB-D 相机 + A100 规划机
  Grounding-DINO → SAM2 → Stage 1 Contact 迁移
  → Stage 2 Motion Field/Goal/Trajectory
  → camera_plan.npz
                         │ 文件传输
                         ▼
连接 Aubo 的机器人电脑
  读取 camera_plan.npz → 手眼变换 → IK/碰撞检查 → Aubo 执行
```

机器人电脑只负责执行已经生成的计划，不需要安装 DINO、SAM2、FGW、GraspNet 或 Stage 2 训练环境。因此 3060/低显存机器也可以作为执行端；DINO/SAM2 权重和 Stage 2 checkpoint 保留在规划机即可。

## 2. 需要发送的文件

从 `deployment_bundle/aubo_camera_execution/` 整个目录复制到机器人电脑。最小必需集合是：

```text
aubo_camera_execution/
├── sample_plan/camera_plan.npz
├── config/handeye.yaml
├── scripts/execute_camera_plan_aubo.py
└── requirements_robot.txt
```

建议一并发送以下复核文件：

```text
sample_plan/strict_inference_report.json  # 推理配置和来源记录
sample_plan/inference_overlay.png        # RGB、检测/热力/轨迹叠加图
sample_plan/camera_grasp_trajectory.ply   # Open3D 可视化点云和夹爪轨迹
bundle_manifest.yaml                      # 文件、数组 shape 和坐标约定
README_zh.md                              # 现场简明说明
```

`camera_plan.npz` 是真正的机器接口，不要只发送 PNG/PLY。其关键数组为：

| 数组 | shape | 含义 |
|---|---:|---|
| `tcp_camera` | `[4,4]` | 相机坐标系下抓取 TCP 位姿 |
| `tcp_trajectory_camera` | `[64,4,4]` | 相机坐标系下 64 步 TCP 轨迹 |
| `object_trajectory_camera` | `[64,4,4]` | 相机坐标系下对象轨迹，供复核使用 |
| `first_contact_camera` / `second_contact_camera` | `[3]` | 抓取接触对 |
| `intrinsic_cv` | `[3,3]` | 生成该计划时使用的相机内参 |

当前样例计划来自已验证的 `strict_motion_memory_corrected_v3`，并且 `tcp_trajectory_camera[0]` 与 `tcp_camera` 一致，避免执行一开始跳向错误的对象坐标系。

## 3. 手眼标定和坐标变换

在机器人电脑上编辑 `config/handeye.yaml`，将占位矩阵替换成标定得到的 **OpenCV 相机坐标系到 Aubo 基座坐标系** 的齐次变换：

```text
T_base_tcp = T_base_camera @ T_camera_tcp
```

LFV 相机坐标约定为：x 向图像右方、y 向图像下方、z 向相机前方。Aubo 驱动若采用不同轴定义，必须在 `AuboAdapter` 内显式转换。不要通过修改 `camera_plan.npz` 来“修正”轴方向。

标定完成后，至少用一个已知棋盘格/标记点验证：相机中正前方的点是否变换到机器人预期方向，单位是否为米，旋转矩阵是否正交且行列式为 +1。

## 4. 安装和 dry-run

机器人电脑建议建立一个干净的 Python 环境，然后安装执行端唯一需要的依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_robot.txt
```

先不要连接真实运动，运行 dry-run 检查变换后的抓取位姿和轨迹首帧：

```bash
python scripts/execute_camera_plan_aubo.py \
  --plan sample_plan/camera_plan.npz \
  --handeye config/handeye.yaml
```

脚本应打印抓取位姿、64 步轨迹数量和首帧位姿，并明确输出 `dry-run: no Aubo commands sent`。如果手眼矩阵仍是单位阵，输出只用于检查格式，不能直接执行。

## 5. 接入 Aubo 并执行

`scripts/execute_camera_plan_aubo.py` 中的 `AuboAdapter` 是唯一的厂商接口边界。根据另一台电脑上已有的 Aubo SDK/控制项目，实现以下方法：

```python
connect()                         # 建立控制连接并清除急停/错误状态
move_linear(pose_base, speed_m_s) # 笛卡尔直线运动，pose_base 为 [4,4]
open_gripper()                    # 张开夹爪
close_gripper()                   # 完全闭合并保持
stop()                            # 异常或结束时停止/断开
```

不要在该适配器中重新实现视觉推理。接入后先以低速、空夹爪和假物体测试。正式执行命令为：

```bash
python scripts/execute_camera_plan_aubo.py \
  --plan /path/to/camera_plan.npz \
  --handeye /path/to/handeye.yaml \
  --execute
```

脚本的固定动作顺序是：

1. 连接机器人并张开夹爪；
2. 沿抓取 TCP 的局部 `-z` 方向退后 `pregrasp_distance_m`，到达预抓取位姿；
3. 用平移线性插值和旋转 Slerp 分 `approach_steps` 步靠近抓取位姿；
4. 等待 `settle_seconds` 后执行完全闭合；
5. 按 `tcp_trajectory_camera` 的 64 个位姿逐步运动。

每个目标位姿送入 Aubo 前仍应由机器人侧执行 IK、关节限位、工作空间、速度/加速度和碰撞检查。任何检查失败都应停止，不要跳过该步继续运行。

## 6. 规划机如何产生新的交付文件

当输入 RGB-D 改变时，新的 `camera_plan.npz` 仍在 LFV 仓库中生成。规划机需要：

```text
LFV 仓库
Grounding-DINO 权重和环境
SAM2 独立 conda 环境及权重
DINOv2 权重
Stage 1 source memory
Stage 2 best.pt
RGB、depth、intrinsics.yaml
```

使用严格单帧入口（示意）：

```bash
PYTHONPATH=. python scripts/deployment/run_strict_camera_inference.py \
  --input-dir /path/to/rgbd_input \
  --output-dir /path/to/inference_output \
  --perception-config configs/pipeline/hand_pouring.yaml \
  --transfer-config configs/affordance_transfer/episode0_to_ace_red_mug_fgw_k64.yaml \
  --sam2-python /path/to/sam2/bin/python \
  --stage2-checkpoint /path/to/stage2/best.pt \
  --motion-memory /path/to/motion_memory.npz
```

该入口按既有逻辑执行 Grounding-DINO 检测、SAM2 分割、Stage 1 Soft Heatmap AffCorrs+FGW 迁移、固定 256 点对齐采样、Stage 2 Motion Field/Goal/Full64 推理和 top-down 接触对实例化。完成后将输出目录中的 `camera_plan.npz` 以及复核文件复制到执行包的 `sample_plan/`（实际运行时建议使用带时间戳的新目录，而不是覆盖样例）。

## 7. 安全边界和故障排查

- 首次实机运行必须使用低速和较大的预抓取距离；确认 TCP 的局部 z 轴确实朝向抓取接近方向。
- `camera_plan.npz` 只描述相机系计划，不包含手眼标定；矩阵错误会使整条轨迹整体偏移。
- 机器人电脑没有 RGB-D 推理能力时，不能把新的图片直接交给执行脚本；必须在规划机重新生成计划后再传输。
- 若 dry-run 的首帧不等于抓取位姿，说明使用了旧计划或错误版本的规划输出，应停止执行并重新生成。
- `--execute` 报 `NotImplementedError` 是预期的 SDK 未接入提示；将真实 Aubo API 只填入 `AuboAdapter`，不要修改坐标变换和轨迹逻辑。

