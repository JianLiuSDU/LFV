# Drawer Open v2：正前方接触迁移、Top-down GraspNet、Full64 与执行闭环

## 1. 本轮纠正后的结论

本轮废弃旧的 `drawer_open_v1/seed_0` 场景结论，固定使用
`/home/users1/ljian/lfv_runs/drawer_open_v2/front_seed_0`。旧版本的抽屉过高、朝向为
`yaw=pi`，推理相机位于机械臂一侧，和真实视频中“相机在桌前、抽屉把手朝相机、
机械臂位于物体后方”的数据分布不一致。v2 将仿真截图、热力迁移、完整点云抓取、
64 步运动推理和执行录像全部放在同一个布局中，并通过 snapshot/execution 的实体位姿
误差检查防止不同场景产物被误混用。最终仿真抓取和开抽屉均成功，执行期间没有丢失
抓取。

## 2. 数据来源与只读隔离

### 2.1 第一阶段接触热力

使用的源样本不是 motion 数据，也不是杯子数据，而是：

- 原始数据：`/media/ljian/lj/hand_data/drawer/episode_60`
- LFV 工作目录：`/media/ljian/lj/data_3d/hand_drawer_lfv/episode_60`
- `episode_60/rgb` 的 `readlink -f` 结果是原始目录的同名 `rgb`，因此原始图片只读，
  新生成的掩码、热力和报告只写入 `data_3d`。
- 源图为 `frame_000045`；anchor frame 为 45；检测到的接触区间为 51–72；实际融合
  帧为 48、51、54、57。
- 源热力使用 4096 点物体点云，深度有效率 0.9698，生成 264 个接触种子。机器可读
  事实位于 `contact_heatmap/contact_heatmap_meta.json`。

源热力由整柜/把手前景、手部遮挡证据和可见物体点云构造；手部遮挡像素在 anchor
物体表面形成连续热力，不把抓取位姿标签混入热力迁移阶段。

### 2.2 第二阶段轨迹与训练

- 原始运动数据：`/media/ljian/lj/new_data/drawer`
- 处理目录：`/media/ljian/lj/data_3d/drawer_lfv_v2`
- 训练视图：`/media/ljian/lj/data_3d/drawer_lfv_v2_train`

例如 `drawer_lfv_v2/episode_0/rgb` 和 `camera_0.mp4` 分别解析回
`new_data/drawer/episode_0` 中的原始文件。DINO、SAM、采样、TAPIP3D、SVD 和审核
结果写在处理目录；训练视图只软连接最终通过质量门的 106 个 episode，并由
`dataset_view.json` 固定 episode 清单和 manifest 路径。原始数据没有被修改。

第二阶段的语义角色始终是：`manipulated = black drawer handle / drawer pull`，
`reference = green drawer cabinet / cabinet housing`。每条 episode 使用 metadata 中的
相机内参和 `depth_scale` 恢复米制点云，采 256 个 manipulated 点和 256 个 reference
点；TAPIP3D 跟踪把手，SVD 压缩为刚体 SE(3)，最后按弧长重采样为 64 步。

## 3. 修正后的场景、相机与坐标系

固定布局为：

```text
世界 +X：抽屉开启方向，也是桌前相机所在方向
世界 -X：Panda 基座所在方向
世界 +Z：向上
drawer root = (-0.06, 0.0, 0.004), yaw = 0
base_camera eye = (0.50, 0.0, 0.52)
base_camera target = (-0.06, 0.0, 0.035)
```

因此 base camera 从桌前正对抽屉和机械臂，抽屉黑色把手朝相机，Panda 位于抽屉后方。
保存的 `rgb_base_camera.png` 既是 Soft Heatmap AffCorrs 的目标输入，也是 Full64 坐标
系叠加使用的背景；执行器另保存同一 base camera 的正前方录像，斜视角只用于检查
抓取深度和抽屉开度。

v2 柜体宽 0.24 m、深 0.28 m、高 0.095 m；把手横杆宽 0.10 m、半厚 0.008 m、
前向伸出 0.032 m、中心高度 0.042 m。这个浅柜比例来自真实 drawer 视频分布，旧版
26 cm 高柜体不再作为本任务基线。

坐标约定：图像反投影使用 OpenCV camera frame（+X 右、+Y 下、+Z 前）；完整点云和
GraspNet 同时保存 camera/world/manipulated-link 三种表示；运动模型预测把手 link 的
世界 SE(3)；执行前用初始 link 位姿把物体系 GraspNet 行转换为 Panda TCP 世界位姿。

## 4. 第一阶段：Soft Heatmap AffCorrs 到完整点云抓取

配置：

- `configs/affordance_transfer/drawer_episode60_to_maniskill_front.yaml`
- `configs/affordance_grasp/drawer_episode60_to_maniskill_front_topdown.yaml`

计算流程：

1. 对 episode 60 frame 45 与修正后的仿真正视图做相同的 bbox 留边、等比例缩放和
   padding，冻结 DINOv2 提取并 L2 归一化稠密 patch 描述符。
2. 源正热力 patch 做热力加权 K-Means；目标前景做过分割；正向区域投票 `V` 与目标
   原型回到源全部前景的热力加权反向验证 `Q` 相乘，得到连续目标热力 `H=VQ`。
3. 目标热力插值回 640×480、乘目标 mask 并归一化。固定输出 2×2 图，依次展示源原图、
   源热力、仿真目标原图和迁移热力。当前 global/cycle/peak/entropy 为
   0.4481/0.2941/1.0000/0.6942，迁移通过置信度门。
4. 利用同步深度将 2D 热力反投影到可见把手；再在完整把手表面按距离、法向对置、
   夹持宽度和局部同表面传播构造可验证的 antipodal 接触对。场景中的整柜和桌面点云
   只作为碰撞上下文，不被赋予把手热力。
5. GraspNet 在 25,600 点输入上生成方向候选，并用接触对细化。最终硬条件同时要求：
   接近方向接近世界 `-Z`、闭合方向接近世界 `+X`、接触弦与闭合轴点积至少 0.9、两点
   近似等高、两指均在热区、接触对中心与指尖中点接近，以及指/掌/下降路径/全局碰撞
   全部低于阈值。

这里的 top-down 含义是夹爪沿世界 `-Z` 从上向下接近，两个指板沿世界 `X` 闭合，
从而横跨把手朝相机的一面和背面；不是沿竖直方向夹住把手上下表面。最终候选没有被
人工平移：接近误差 0.148°，闭合方向误差 4.696°，前后接触宽度 16.054 mm，接触弦
与闭合轴点积 1.000，两指热力 0.869/0.876，接触两点世界高度差 0.041 mm，所有严格
碰撞 IoU 均为 0。

此前曾把 GraspNet 候选强制平移到把手中央；诊断发现中央最近接触对实际连接上下表面，
与闭合轴点积不足 0.1。现在 `min_pair_closing_alignment` 与
`max_nearest_pair_distance` 是硬约束，筛选失败会输出逐候选 diagnostics，禁止再次把
视觉上像 top-down、物理上却没有跨过前后两面的姿态当作正确抓取。

## 5. 第二阶段模型与 64 步推理

正式 checkpoint：

- Goal Pose：`epoch=0800-val_sample_goal_pos_err_cm=2.115.ckpt`
- Full64：`epoch=1400.ckpt`

Goal Pose 使用 256 点把手、256 点整柜 reference、机器人状态和语言 embedding，预测
终点 3D 平移 + 6D 连续旋转；Full64 使用 64 点条件和 Goal Pose，输出 64 个相对起始
把手位姿 `[x,y,z,qx,qy,qz,qw]`。二者推理均使用 checkpoint 的 EMA 与 normalizer。

修正场景上的原始网络预测从初始世界 x=-0.06 m 移动到 x=0.06459 m，主要方向是世界
`+X`，即朝桌前相机拉出，原始位移约 12.46 cm；末态相对旋转约 10.80°。原始 64 步
完整保存在 NPZ 和坐标系叠加图中。执行层知道 drawer 是 prismatic joint，因此只在
执行时将轨迹投影到单调 `+X` 轴并把安全拉距限制为 8.5 cm；这不改写网络输出。

## 6. Top-down 可达性与执行控制

杯子使用的 `panda_long_finger` 宽 30 mm；它在热力峰值一侧下降时会碰到 U 形把手的
端部支撑。drawer 因此新增同接口的 `panda_drawer_finger`：指板长 70 mm、宽 16 mm、
厚 4.5 mm、高摩擦，接触面积仍是 stock Panda pad 的 3.46 倍；指板相对 TCP 沿手指轴
再下移 30 mm，使机械臂在柜体顶面之上时，指板下段仍能包住低位把手。

另一个关键点是下降时不能保持 80 mm 全开：内侧指板会先落到柜体顶面。执行接口新增
`approach_gripper_action`，本任务固定为 0.0，对应约 30 mm 总开口；动作顺序为：

```text
初始全开保持
→ 移动到预抓取并预成形为约 30 mm
→ 以预成形开口 top-down 下降到前后间隙
→ 到位后持续发送 -1.0 完全闭合
→ 闭合保持并确认 is_grasped
→ 跟随经过安全投影的 Full64
→ 末态保持
```

`tools/calibrate_drawer_grasp_reachability.py` 固定物体系 GraspNet 行，只扫描 TCP 局部
接近轴偏移。`-0.055 m` 和 `-0.060 m` 均能夹住把手，正式配置使用 `-0.060 m`；闭合后
两指 qpos 为 7.653/7.626 mm，表明夹爪受把手阻挡而非闭合到底抓空。

正式执行结果：snapshot/execution 初始把手对齐误差 0.00000006 m；完全闭合后
`is_grasped=true`；501 帧中没有丢失抓取；末态仍为 `is_grasped=true`；物理 drawer
qpos=0.08540 m；仿真 `success=true`；抽屉末态位置误差 0.00040 m。正视视频为
640×480@30fps、斜视视频为 1280×720@30fps，均为 501 帧/16.7 秒。

## 7. 固定产物

全部权威产物位于 `/home/users1/ljian/lfv_runs/drawer_open_v2/front_seed_0`：

- 仿真正视图：`snapshot/rgb_base_camera.png`
- 源/目标 2×2 热力迁移：`affordance_transfer/transfer_source_target_2x2.png`
- 迁移完整诊断：`affordance_transfer/transfer_summary.png`
- Open3D 完整点云 + 抓取：`grasp/graspnet_selected_open3d.png`
- 抓取四联报告：`grasp/topdown_grasp_summary.png`
- 64 步坐标系：`motion_inference/full64_coordinate_frames_overlay.png`
- 正前方录像：`execution/drawer_open_execution_front.mp4`
- 斜视角录像：`execution/drawer_open_execution.mp4`
- 执行报告：`execution/execution_report.json`
- 最终标定报告：`topdown_reachability_drawer_finger_final.json`

JSON 报告是数值事实来源，PNG/MP4 是快速迭代中固定的人工检查基础设施。

## 8. 代码复用边界与复现

通用模块仍与 pouring 共用：`lfv/affordance_transfer`、`lfv/lifting`、`lfv/geometry`、
`scripts/sim/generate_graspnet_from_full_contact.py`、
`scripts/inference/infer_functional_motion.py`、
`scripts/robot/execute_functional_motion_maniskill.py`。drawer 的差异只在任务环境、查询词、
方向/碰撞配置和 gripper 规格中。

完整配置：

`configs/experiments/functional_motion/drawer_open_episode60_front_topdown_execution.yaml`

重跑命令：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/maniskill3/bin/python \
  scripts/run_functional_motion_execution.py \
  --config configs/experiments/functional_motion/drawer_open_episode60_front_topdown_execution.yaml
```

各阶段可用 `--skip-snapshot`、`--skip-transfer`、`--skip-grasp`、`--skip-motion`、
`--skip-execution` 独立迭代。抓取改变后必须重新执行 reachability calibration；场景布局
改变后必须重建 snapshot、重新迁移并重新推理，不允许沿用旧布局的抓取或轨迹。

## 9. Handle-only mask 消融实验

为判断整柜前景是否导致中心接触扩散，保持 episode 60 frame 45、源连续热力、DINOv2
权重、6/64 聚类数、温度和置信度配置不变，只替换两侧前景 mask：源侧用
GroundingDINO `black drawer handle / drawer pull handle` 加 SAM2 生成的
`handle_sam_mask/handle_mask.npy`（SAM2 score 0.8789），目标侧使用 snapshot 已有的
`manipulated_mask`。本实验只运行二维迁移，不运行深度提升、点云补全或 GraspNet。

配置与输出：

- mask 配置：`configs/pipeline/hand_drawer_handle_mask_ablation.yaml`
- 迁移配置：`configs/affordance_transfer/drawer_episode60_handle_only_to_maniskill_front.yaml`
- 输出：`lfv_runs/drawer_open_v2/front_seed_0/affordance_transfer_handle_only`

对比整柜 mask，handle-only 将目标把手外的热力质量从 681.09 降为 0，把手中央三分之一
的热力占比从 30.64% 提高到 39.27%，`heat>0.5` 的把手覆盖率从 94.37% 降为
61.89%。但输出仍覆盖横杆大部，global confidence 从 0.4481 降为 0.1413，置信度熵
从 0.6942 升为 0.9718。结论是：仅换 handle mask 可以消除柜体泄漏并略微增强中央，
但会失去整柜上下文，且 DINO 区域聚类仍不能稳定编码“把手内部横向位置”，因此不能
单独作为最终的中心接触迁移方法。

## 10. 目标聚类数量与把手中心聚焦消融

在 handle-only 实验上进一步只改变目标侧 K-Means 数量，固定源 episode 60 frame 45、
源聚类数 6、两侧 mask、DINOv2 特征、连续源热力、温度和置信度计算，分别测试目标
`K=8/16/32/64/96/128/160`。该消融仍然只做二维热力迁移，不运行点云提升、GraspNet
或执行。统一脚本为 `scripts/affordance_transfer/sweep_target_clusters.py`，全部结果保存在：

`lfv_runs/drawer_open_v2/front_seed_0/affordance_transfer_handle_only_cluster_sweep`

| 目标 K | global confidence | 中央 1/3 热力质量 | `heat>0.5` 覆盖率 | 像素热力熵 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0928 | 36.55% | 74.96% | 0.9821 |
| 16 | 0.1216 | 36.73% | 74.65% | 0.9830 |
| 32 | 0.1364 | 38.67% | 64.24% | 0.9813 |
| 64 | 0.1413 | 39.27% | 61.89% | 0.9806 |
| 96 | 0.1416 | 39.69% | 62.75% | 0.9799 |
| 128 | 0.1397 | 40.16% | 61.19% | 0.9797 |
| 160 | 0.1423 | 40.69% | 58.69% | 0.9799 |

增大 K 确实会减小每个目标聚类广播热力的区域：中央三分之一质量从 K=8 的 36.55%
提高到 K=160 的 40.69%，高热区域覆盖率从 74.96% 降到 58.69%。但收益在 K=64
之后明显趋于饱和，像素熵始终约为 0.98，峰值位置还会随 K 在把手左右 patch 间跳动；
因此 K=128 或 K=160 可作为当前更聚焦的工程设置，不能把“增加聚类数”视为中心定位
问题的根本解法。原因是 K-Means 只根据 DINO 语义特征分区，不显式知道把手内部的
相对横向坐标，也不保证一个聚类在图像上空间连通；黑色横杆不同位置的描述符非常相似，
正反向 Softmax 仍会给许多 patch 相近分数。下一步若要求稳定单峰，应优先比较逐 patch
匹配（取消目标 K-Means 广播）、空间连通聚类，以及把手归一化局部坐标作为弱位置条件，
而不是继续把 K 增大到接近 patch 数量。

复现实验：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/maniskill3/bin/python \
  scripts/affordance_transfer/sweep_target_clusters.py \
  --config configs/affordance_transfer/drawer_episode60_handle_only_to_maniskill_front.yaml \
  --output-root /home/users1/ljian/lfv_runs/drawer_open_v2/front_seed_0/affordance_transfer_handle_only_cluster_sweep \
  --target-clusters 8 16 32 64 96 128 160 \
  --device cuda
```

人工对比入口为 `cluster_sweep_comparison.png`，完整数值和每个 K 的产物路径记录在
`cluster_sweep_report.json`。

## 11. Handle-only K=160 完整抓取与执行闭环

聚类消融之后建立了独立实验目录
`lfv_runs/drawer_open_v2/front_seed_0_handle_only_k160`，没有覆盖第 7 节的整柜 mask
基线。该实验从同一仿真布局重新导出 snapshot，使用 episode 60 frame 45 的把手 mask
和连续热力，目标侧 K=160 完成二维迁移，再依次运行完整表面提升、top-down GraspNet、
Goal Pose + Full64 推理和 ManiSkill 执行。

对应配置：

- 迁移：`configs/affordance_transfer/drawer_episode60_handle_only_k160_to_maniskill_front.yaml`
- 抓取：`configs/affordance_grasp/drawer_episode60_handle_only_k160_to_maniskill_front_topdown.yaml`
- 完整执行：`configs/experiments/functional_motion/drawer_open_episode60_handle_only_k160_execution.yaml`

K=160 迁移被置信度门控接受，global confidence 为 0.1423。最终 GraspNet 接触对位于
把手物体系 `y=+6.65 mm`，相比旧基线的 `y=-29 mm` 明显靠近横杆中心；接触高度约
49.09 mm，前后接触宽度 16.01 mm。抓取接近方向误差 0.040°、闭合方向误差 1.454°、
接触弦与闭合轴对齐度 1.0、两指热力 0.903/0.896，global/finger/palm/approach-path
严格碰撞 IoU 全部为 0。

执行闭合完成即 `is_grasped=true`，501 帧中没有丢失抓取，末态仍保持抓取；drawer
qpos=0.08540 m，仿真 `success=true`，末态位置误差 0.00040 m。最终斜视视频为
1280×720@30fps，正视视频为 640×480@30fps，均为 501 帧、16.7 秒。固定检查产物：

- 迁移四联图：`affordance_transfer/transfer_source_target_2x2.png`
- 抓取报告：`grasp/topdown_grasp_summary.png`
- Open3D 抓取：`grasp/graspnet_selected_open3d.png`
- 64 步坐标系：`motion_inference/full64_coordinate_frames_overlay.png`
- 斜视视频：`execution/drawer_open_execution.mp4`
- 正视视频：`execution/drawer_open_execution_front.mp4`
- 数值报告：`execution/execution_report.json`

完整复现命令：

```bash
cd /home/users1/ljian/LFV
/home/users1/ljian/anaconda3/envs/maniskill3/bin/python \
  scripts/run_functional_motion_execution.py \
  --config configs/experiments/functional_motion/drawer_open_episode60_handle_only_k160_execution.yaml
```
