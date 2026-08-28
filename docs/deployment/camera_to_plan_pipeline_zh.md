# 单张 RGB-D 到相机坐标抓取与轨迹的离线部署闭环

本目录新增的部署入口用于把 A100 上的 LFV 推理结果交给另一台只负责控制 Aubo 的电脑。输入固定为一个文件夹（默认 `/home/users1/ljian/LFV_ex/cup_pouring/ex_1/input`），输出固定写入其父目录（默认 `ex_1`）。部署脚本不直接连接机械臂，也不假设手眼标定矩阵；所有输出都在 OpenCV 相机坐标系，第二台电脑只需使用自己的手眼标定把矩阵变换到机器人基座。

## 输入契约

`rgb.png`（或 `rgb.jpg/color.png`）、`depth.npy`/`depth.png`（米；可在 `manifest.json` 中设置 `depth_scale`）以及 `intrinsics.json`。内参支持 `intrinsic_cv`/`matrix`/`K` 或 `fx,fy,cx,cy`。检测分割后端必须提供同尺寸的 `cup_mask` 和 `bowl_mask`（PNG/NPY，非零为前景）。推荐将检测器和 SAM/SAM2 封装为外部命令，命令写出 `masks.npz`，从而不把特定显卡依赖混入 LFV。

## 计算流程

1. 读取并检查 RGB-D、内参和两个前景 mask。
2. 调用可插拔的 AffCorrs/FGW 外部后端，输入源 episode 的 RGB、mask、连续热力图和当前目标 RGB/mask，输出 `target_heatmap[H,W]`。当前仓库也提供 `PrecomputedHeatBackend`，用于复现实验中已经在 A100 生成的热力图。
3. 对杯子 mask 的有效深度反投影到 OpenCV 相机坐标。`VisibleOnlyCompletionBackend` 只用于 smoke test；正式运行应使用 `SAM3DSubprocessBackend` 或 A100 离线产生的 `NPZCompletionBackend`。SAM3D 的 canonical mesh 不被假定为相机坐标，必须由外部配准后写入 `complete_points_camera` 和 `camera_from_object`。
4. 将可见热力通过最近邻从可见点传到补全点，保持连续值，不做二值化。该步骤只负责把语义热力附着到几何表面；真正的隐蔽面传播仍由补全/配准后端负责。
5. GraspNet 后端读入完整杯子点云和 3D 热力，按检测器分数、热力和可配置的相机“上方”方向筛选候选，输出 `selected_grasp_camera[4,4]`。推荐 `grasps.npz`（`[M,17]` GraspNet 行格式）或外部命令；几何 fallback 只能用于调试，报告会明确标注不是 GraspNet。
6. 运动后端可调用已有的双 checkpoint pouring 模型。`LegacyPouringBackend` 通过临时的 world==camera snapshot 复用原脚本，并将 goal 和 object trajectory 统一转换为相机坐标。新模型可实现同样的 `MotionPrediction` 接口而不改下游。
7. 用选定 grasp 的 attachment 将 object trajectory 转成 TCP trajectory，保存 NPZ、JSON 报告、2D 相机叠加图和 Open3D 截图（若当前环境安装了 Open3D）。

## 输出文件

`camera_plan.npz` 包含 `selected_grasp_camera[4,4]`、`object_trajectory_camera[T,4,4]`、`tcp_trajectory_camera[T,4,4]`、`complete_points_camera[N,3]` 和 `complete_heat[N]`；`target_heatmap.npy`、`camera_overlay.png`、`open3d_snapshot.png` 与 `camera_plan_report.json` 用于快速判断。`open3d_unavailable.txt` 表示仅缺少可视化环境，不会伪造截图。

## 运行

```bash
conda run -n lfv_open3d_vis python scripts/deployment/run_camera_to_plan.py \
  --config configs/deployment/cup_pouring_camera.yaml
```

正式运行前应把 config 的 transfer、completion、grasp、motion 后端替换为实际 artifact/命令。没有 SAM3D 权重、分割 mask、热力图或 GraspNet artifact 时，程序会在对应步骤明确失败；只有显式设置 `allow_fallback_grasp: true` 才会生成调试用几何夹爪，不会误称为完整闭环。

## 坐标系和交接

相机坐标采用 OpenCV 约定：`x` 向右、`y` 向下、`z` 沿光轴向前，所有平移单位米，姿态为右手 4×4 齐次矩阵。Aubo 电脑读取 `camera_plan.npz`，用手眼标定的 `T_base_camera` 左乘每个 TCP pose，并在执行前重新做 IK、工作空间和碰撞检查；本仓库不保存或猜测任何真实相机外参。
