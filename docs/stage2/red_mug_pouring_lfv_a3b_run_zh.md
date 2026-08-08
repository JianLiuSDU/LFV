# 红色杯子 `pouring_lfv` A3b 仿真推理记录

## 1. 本次验证范围

本次实验固定使用由 `pouring_lfv` 数据训练得到的 A3b 模型，不替换网络、不使用旧 `pouring` 模型，也不对预测轨迹做人工终点修正。完整链路为：仿真红色带把手杯场景快照 → AffCorrs+FGW 热力迁移 → 完整点云热力提升 → GraspNet 方向/碰撞约束抓取 → A3b Goal Diffusion → A3b Full64 Trajectory Diffusion → ManiSkill 长指平行夹爪执行与双视角录像。

运行配置为：

- 总流程：`configs/stage2/red_mug_pouring_a3b_execution.yaml`
- 热力迁移：`configs/affordance_transfer/episode0_to_ace_red_mug_fgw_k64.yaml`
- 抓取生成：`configs/affordance_grasp/episode0_to_ace_red_mug_fgw_k64_topdown.yaml`
- A3b 检查点：`lfv_runs/stage2/ablation_stage_aware/a3b_gated_phase_tokens/checkpoints/best.pt`（epoch 141，EMA 权重）
- 红杯资产：`ACE_Coffee_Mug_Kristen_16_oz_cup_scale`
- 源热力：`/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0` 的第 39 帧

## 2. 固定输入和采样合同

场景中红杯位于 `(x=0.04, y=-0.12, yaw=-pi/2)`，碗位于 `(x=0.06, y=0.18)`，杯把朝相机左侧并保持与演示图像相近的可见性。Stage2 输入为红杯与碗各 256 个可见点；每个点的 XYZ 与 384 维 DINOv2 特征使用完全相同的像素采样索引。模型推理采样 16 个终态，每个终态采样 2 条 Full64 轨迹，两个扩散解码器均使用 50 个 DDIM 步。

## 3. 第一阶段与抓取结果

AffCorrs+FGW 迁移被置信度门控接受：全局置信度为 `0.4187`，保留热量为 `0.8976`。热力集中于红杯左侧把手上半部，没有扩散到杯身。

GraspNet 在完整点云上先生成候选，再应用 top-down、双指热力、把手两侧接触、指尖表面距离和碰撞约束。最终抓取指标为：

- top-down 偏角：`1.043 deg`
- 双侧物理接触间距：`17.993 mm`
- 两接触点高度差：`0.327 mm`
- 双指热力：`0.898 / 0.898`
- 指尖平均表面距离：`5.0 mm`
- 全局、手指、手掌和接近方向碰撞 IoU：均为 `0.000`

仿真闭合后 `grasp_acquired_after_close=true`，运动结束时 `grasped_at_end=true`，全视频没有掉杯。因此热力迁移、GraspNet 抓取和长指夹爪执行均通过本次验证。

## 4. A3b 轨迹结果与当前结论

被选择轨迹对模型自身终态的拟合是正常的：Full64 终点到 Goal Diffusion 终态的平移误差为 `7.75 mm`、旋转误差为 `1.93 deg`；实际 TCP 最终位置到预测位置误差为 `9.19 mm`，平均 TCP 跟踪误差为 `5.54 mm`。这说明轨迹扩散输出、坐标转换和执行器跟踪是连通的。

但是，Goal Diffusion 预测的世界坐标终点为 `[-0.0683, 0.0230, 0.2104] m`，距碗可见中心 `0.2663 m`（平面距离 `0.2003 m`），所以杯子没有移动到碗正上方，`simulator_success=false`。可视化在执行前已经显示终态位于图像上方；真实执行也复现了相同结果。本次结论必须分开理解：**红杯抓取稳定成功，但 `pouring_lfv` A3b 的目标关系在该仿真场景上泛化失败**。这不是抓取、控制跟踪或轨迹末端未收敛导致的，也没有用后处理隐藏这一问题。

## 5. 输出文件

所有结果位于：

`/home/users1/ljian/lfv_runs/stage2/pouring_lfv_v1/red_mug_a3b_seed_0`

关键输出：

- 热力迁移对比：`transfer/transfer_summary.png`
- 抓取总览：`grasp/topdown_grasp_summary.png`
- Open3D 抓取：`grasp/graspnet_selected_open3d.png`
- 仿真 RGB 抓取：`grasp/graspnet_selected_rgb_clean.png`
- 64 帧坐标系轨迹：`motion_inference/full64_coordinate_frames_overlay.png`
- 终态候选：`motion_inference/goal_pose_candidates_overlay.png`
- 推理总览：`motion_inference/simulation_inference_summary.png`
- 主视角录像：`execution/pouring_execution.mp4`
- 正前方录像：`execution/pouring_execution_front.mp4`
- 执行报告：`execution/execution_report.json`

两路录像均为 30 FPS、348 帧、时长 11.6 秒；主视角为 1280×720，正前方为 640×480。

## 6. 可复现命令

在 LFV 根目录运行完整流程：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/run_pouring_motion_execution.py \
  --config configs/stage2/red_mug_pouring_a3b_execution.yaml
```

快速迭代时可以分别用 `--skip-snapshot`、`--skip-transfer`、`--skip-grasp`、`--skip-motion` 和 `--skip-execution` 复用已有中间结果。总流程脚本现在会在迁移之后自动调用配置指定的 GraspNet 阶段，因此更换同类别杯子时只需新增资产与路径配置，不需要复制执行逻辑。
