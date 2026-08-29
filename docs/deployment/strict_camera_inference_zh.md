# 严格相机输入推理接口

`scripts/deployment/run_strict_camera_inference.py` 是当前仓库的正式单帧入口。它从一个文件夹读取 RGB-D 和相机内参，按训练/实验中已经固定的模块顺序生成相机坐标系抓取位姿和对象轨迹：

```text
RGB-D + intrinsics.yaml
        │
        ├─ lfv.pipeline.dino_bbox
        │    Grounding-DINO 检测 cup / bowl，保存两个 bbox
        ├─ lfv.pipeline.sam2_mask
        │    SAM2 以 bbox 为提示，分别保存 cup / bowl mask
        ├─ lfv.affordance_transfer.app.run_transfer
        │    既有 Soft Heatmap AffCorrs + FGW，迁移 hand_pouring_lfv/episode_0 的 Contact Field
        ├─ lfv.inference.functional_motion.two_stage_pouring
        │    既有采样器：操作物体 256 点、参考物体 256 点，并保持点/DINO 对齐
        ├─ lfv.deployment.model_backend.FunctionalMotionDirectBackend
        │    当前 Stage 2 joint functional-motion checkpoint（XYZ+DINO、三 token、Goal/Full64）
        └─ contact-pair grasp instantiation
             top-down、跨接触区域两侧的相机系 TCP 抓取姿态
```

因此，该入口不使用 GrabCut、手工 ROI、合成 mask、随机点云或旧的语言条件 legacy 模型。检测、分割、Stage 1 迁移和 Stage 2 输入采样都直接调用仓库中原有实现；最后的接触对选择只是将连续 Contact 热力实例化为可执行的 top-down 平行夹爪候选，不改变训练模型。

## 输入目录

目录至少包含：

```text
input/
├── rgb.png                 # 任意 RGB 文件名均可
├── depth.png               # 16-bit 深度，和 RGB 对齐
└── intrinsics.yaml         # 见下例
```

```yaml
camera:
  color:
    camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  depth:
    depth_scale_m_per_unit: 0.001
```

深度会先乘以 `depth_scale_m_per_unit` 转成米，再用 OpenCV 相机内参反投影。RGB 和 depth 必须同分辨率且像素对齐；手眼标定不在此入口中完成。

## 运行

先把 Grounding-DINO、SAM2 和 DINOv2 权重准备在目标机器上，或把配置/命令行参数指向本地路径：

```bash
cd /path/to/LFV_stage2_motion_field
PYTHONPATH=. python scripts/deployment/run_strict_camera_inference.py \
  --input-dir /path/to/input \
  --output-dir /path/to/inference \
  --perception-config configs/pipeline/hand_pouring.yaml \
  --transfer-config configs/affordance_transfer/episode0_to_ace_red_mug_fgw_k64.yaml \
  --sam2-root /path/to/sam2 \
  --sam2-python /path/to/conda/envs/sam2/bin/python \
  --sam2-device cuda:0 \
  --stage2-checkpoint /path/to/stage2/best.pt \
  --dino-weights /path/to/dinov2_vits14_pretrain.pth \
  --device cuda:0 \
  --stage2-device cuda:0
```

没有可用的 Grounding-DINO/SAM2 权重时脚本会直接报错，不会悄悄改用人工分割。SAM2 在 `--sam2-python` 指定的独立 conda 环境中作为子进程运行，主环境不需要安装 `iopath`。`perception-config` 中的 `objects.affordance.prompt` 和 `objects.target.prompt` 分别控制两个检测目标；换任务时只替换配置和 Stage 1 source memory，不修改 pipeline 代码。

若 RGB-D 掩码内部存在明显深度断层，FGW 的原有局部图可能无法连通。默认仍严格使用迁移配置中的阈值；只有在确认是传感器深度断层时，才可以显式传入已有参数覆盖，例如 `--fgw-edge-length-ratio 12`，该值会写入报告，便于复现实验。

## 输出

```text
inference/
├── cup_bbox.npy, bowl_bbox.npy
├── cup_mask.png, bowl_mask.png
├── detection_segmentation_overlay.png
├── camera_snapshot.npz
├── stage1_transfer/
│   ├── transfer_result.npz
│   ├── transfer_summary.png
│   └── transfer_report.json
├── stage2_motion/
│   └── motion_field.npz
├── motion_prediction.npz
├── camera_plan.npz
├── camera_grasp_trajectory.ply
├── inference_overlay.png
└── strict_inference_report.json
```

`camera_plan.npz` 是执行端的主要接口：

* `tcp_camera`: `[4,4]` 相机坐标系 TCP 抓取位姿；
* `tcp_trajectory_camera`: `[64,4,4]` Stage 2 Full64 TCP 轨迹；
* `object_trajectory_camera`: `[64,4,4]` 对象轨迹；
* `first_contact_camera`、`second_contact_camera`: 接触对两侧点；
* `intrinsic_cv`: 本次相机内参；
* `manipulated_points_stage1`、`manipulated_heat_stage1`: 256 点 Stage 1 证据。

`motion_field.npz` 保存 Stage 2 encoder 输出的操作物体/参考物体 motion field，便于检查功能区域是否集中。PNG 和 PLY 用于人工复核，不直接控制机械臂。

需要区分两个“迁移”：本入口中的 Stage 1 `run_transfer` 明确迁移的是源示范的 Contact Field；Stage 2 checkpoint 中的 Motion Functional Field 则由其 relevance head 根据当前目标的 XYZ–DINO 在线预测并保存。当前仓库没有一个独立的“源 Motion Field→目标 Motion Field”FGW 接口，因此这里不会把 Contact 迁移结果误称为运动场迁移，也不会人为拼接一个不存在的先验。

## 机器人电脑如何使用

Aubo/其他机器人电脑只需要接收 `camera_plan.npz`，无需运行 Grounding-DINO、SAM2 或 Stage 2 网络。执行端使用手眼标定得到的相机到机器人基座变换：

```text
T_robot_tcp = T_robot_camera · T_camera_tcp
```

然后对抓取位姿和 64 步轨迹逐帧执行 IK、关节限位、碰撞检查、速度/加速度约束和夹爪开合。`tcp_camera` 的坐标约定为 OpenCV 相机系（x 向右、y 向下、z 向前）；若机器人驱动使用其他轴约定，必须在执行端显式转换，不能修改保存的模型输出。

## 当前实现边界

本入口已经严格复用：

1. `lfv.pipeline.dino_bbox` 的 Grounding-DINO 检测；
2. `lfv.pipeline.sam2_mask` 的 SAM2 box 分割；
3. `lfv.affordance_transfer.app.run_transfer` 的 AffCorrs+FGW Contact 迁移；
4. `sample_heat_point_cloud` / `sample_mask_point_cloud` 的固定 256 点采样；
5. `FunctionalMotionDirectBackend` 加载的当前 Stage 2 joint checkpoint、DINOv2 逐点特征、三 token encoder、Goal diffusion 和 Full64 trajectory diffusion。

当前服务器若缺少 `iopath`、兼容版本的 `transformers` 或 Grounding-DINO 本地权重，无法完成正式 strict run；这属于环境/权重问题，不应以 GrabCut 或旧模型结果冒充正式推理。可先运行 `python -m py_compile` 做代码检查，准备好依赖和权重后再执行上述命令。
