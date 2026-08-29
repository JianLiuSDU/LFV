# 严格复用既有 LFV 推理流程

入口脚本：

```text
scripts/deployment/run_strict_camera_inference.py
```

该脚本不使用 GrabCut、手工 mask 或合成点云 fallback。它严格串联已有模块：

```text
RGB-D + 内参
   │
   ├─ lfv.pipeline.dino_bbox
   │    Grounding-DINO 检测 cup / bowl，输出检测框
   │
   ├─ lfv.pipeline.sam2_mask
   │    SAM2 根据检测框分别分割 cup / bowl
   │
   ├─ lfv.affordance_transfer.app.run_transfer
   │    AffCorrs + FGW，源为 hand_pouring_lfv/episode_0 frame 39
   │
   ├─ lfv.inference.functional_motion.two_stage_pouring
   │    sample_heat_point_cloud(..., 256)
   │    sample_mask_point_cloud(..., 256)
   │    Stage 2 使用原逻辑取前 64 点
   │
   ├─ 已训练 pouring goal checkpoint
   ├─ 已训练 Full64 trajectory checkpoint
   │
   └─ top-down 接触对实例化 → camera-frame TCP + trajectory
```

## 运行环境

推荐使用已经包含 DINOv2、Grounding-DINO、SAM2 和 PyTorch 的环境，例如：

```text
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python
```

需要在执行电脑准备：

1. LFV 代码仓库；
2. Grounding-DINO 模型 `IDEA-Research/grounding-dino-base`（可通过 `object.model_id` 改成已下载的本地目录）；
3. SAM2 源码目录，例如 `/home/users1/ljian/sam2`；
4. SAM2 权重：`/home/users1/ljian/sam2/checkpoints/sam2.1_hiera_large.pt`；
5. DINOv2 权重：`third_party/dinov2_weights/dinov2_vits14_pretrain.pth`；
6. `hand_pouring_lfv/episode_0` 源示范数据和 `contact_heatmap`；
7. Stage 2 的 goal/trajectory checkpoint 和语言 embedding。

Grounding-DINO 和 SAM2 没有可用权重时，严格入口会直接报错，不会偷偷退回到人工 ROI。这样可以避免把不符合论文/实验协议的 mask 当成正式结果。

## 在服务器上运行

```bash
cd /home/users1/ljian/LFV_stage2_motion_field

PYTHONPATH=. \
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
scripts/deployment/run_strict_camera_inference.py \
  --input-dir /home/users1/ljian/LFV_ex/cup_pouring/ex_1/input \
  --output-dir /home/users1/ljian/LFV_ex/cup_pouring/ex_1/strict_inference \
  --device cpu \
  --stage2-device cpu
```

有 GPU 时可分别改成 `--device cuda:0 --stage2-device cuda:0`。脚本会自动读取：

* RGB 图像；
* 16-bit 深度图，并按 `intrinsics.yaml` 中的 `depth_scale_m_per_unit` 转成米；
* YAML 中的 color camera matrix。

## 输出文件

```text
strict_inference/
├── cup_bbox.npy
├── bowl_bbox.npy
├── cup_mask.png
├── bowl_mask.png
├── detection_segmentation_overlay.png
├── camera_snapshot.npz
├── stage1_transfer/
│   ├── transfer_result.npz
│   ├── transfer_summary.png
│   └── transfer_report.json
├── stage2_motion/
│   └── legacy_motion/pouring_motion_prediction.npz
├── motion_prediction.npz
├── camera_plan.npz
├── camera_grasp_trajectory.ply
├── inference_overlay.png
└── strict_inference_report.json
```

机械臂执行端主要读取 `camera_plan.npz`：

* `tcp_camera`：抓取 TCP，相机坐标系；
* `tcp_trajectory_camera`：64 步 TCP 轨迹，相机坐标系；
* `object_trajectory_camera`：对象轨迹；
* `intrinsic_cv`：本次相机内参。

执行前必须使用目标机器人的手眼标定矩阵完成：

```text
T_robot_tcp = T_robot_camera · T_camera_tcp
```

然后再做 Aubo IK、关节限位、碰撞检查和速度规划。`camera_grasp_trajectory.ply` 和 PNG 仅用于复核，不直接控制机械臂。

## 配置替换

如果换相机，只需要保证：

* RGB 和 depth 分辨率一致并且深度对齐到 color；
* 更新 `intrinsics.yaml`；
* 修改 `--input-dir`；
* 将 Grounding-DINO/SAM2/DINOv2 权重路径改为新电脑的路径；
* 将 Stage 2 checkpoint 和 `lang_emb.npy` 改为新电脑路径。

推理算法本身不需要改动。若新机械臂使用 Aubo，只需在执行端把相机坐标系轨迹转换为 Aubo 基座坐标系，并实现 Aubo 的 IK/轨迹发送接口。

