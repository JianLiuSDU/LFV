# LFV 相机计划交给 Aubo 执行

本目录是从 A100 规划机发送到机器人电脑的最小执行包。`camera_plan.npz` 已经包含相机坐标系下的抓取 TCP 和 64 步 TCP 轨迹；机器人电脑不需要运行 DINO、SAM2、FGW 或扩散模型。

## 发送内容

至少发送：

```text
camera_plan.npz
config/handeye.yaml
scripts/execute_camera_plan_aubo.py
```

建议同时发送 `strict_inference_report.json`、`inference_overlay.png` 和 `camera_grasp_trajectory.ply`，用于执行前人工复核。示例文件位于 `sample_plan/`。

## 坐标和执行顺序

LFV 输出使用 OpenCV 相机系：x 向右、y 向下、z 向前。将每个位姿左乘手眼矩阵：

```text
T_base_tcp = T_base_camera @ T_camera_tcp
```

脚本会在发出任何机器人指令前检查必需数组、齐次矩阵、有限数值，以及
`tcp_trajectory_camera[0] == tcp_camera`。执行顺序固定为：

1. 读取并检查 `camera_plan.npz`；
2. 打开夹爪；
3. 沿 TCP 的 -z 方向移动到预抓取位姿；
4. 插值到 `tcp_camera` 对应的抓取位姿；
5. 完全闭合夹爪并等待稳定；
6. 按 `tcp_trajectory_camera` 的 64 个位姿逐步执行；
7. 每一步都执行 Aubo IK、关节限位、速度限制和碰撞检查。

## 运行方式

先在 `config/handeye.yaml` 填入真实的 `T_base_camera`，再进行 dry-run：

```bash
python scripts/execute_camera_plan_aubo.py \
  --plan sample_plan/camera_plan.npz \
  --handeye config/handeye.yaml
```

确认位置和姿态正确后，在 `execute_camera_plan_aubo.py` 的 `AuboAdapter` 中接入 Aubo SDK 的连接、直线运动、开夹爪和闭夹爪函数，再使用：

```bash
python scripts/execute_camera_plan_aubo.py \
  --plan /path/to/camera_plan.npz \
  --handeye /path/to/handeye.yaml \
  --execute
```

默认脚本不会伪造 Aubo SDK 调用；未实现适配器时 `--execute` 会明确报错。首次实机测试必须降低速度，并先使用空夹爪/假物体验证手眼方向、TCP 轴方向和工作空间。

## A100 规划机和机器人电脑的边界

规划机需要完整 LFV 仓库、Grounding-DINO、SAM2 环境、DINOv2 权重、Stage1 source memory 和 Stage2 checkpoint。机器人电脑只接收最终 `camera_plan.npz`，因此可以使用低显存机器。若要在机器人电脑现场重新规划，则必须额外部署完整推理环境和 RGB-D 相机驱动。
