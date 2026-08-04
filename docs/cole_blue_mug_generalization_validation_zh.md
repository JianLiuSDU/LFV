# 新杯子实例同类别泛化验证：Cole Hardware Blue Mug

更新日期：2026-07-31

## 1. 验证问题

本实验只改变目标杯子实例，验证同一个 `episode_0` 源接触热力是否还能：

1. 在新杯子的相机图像中迁移到把手，而不是杯身；
2. 提升到新杯子的完整表面并形成反平行接触对；
3. 生成接近世界竖直向下、跨把手两侧且通过静态碰撞检查的抓取。

正式对比统一使用随机种子 0、相同源帧、相同 DINOv2 权重、相同相机、相同
杯碗位置和相同抓取筛选参数。改变的只有杯子视觉与几何资产。新实例为本地扫描
资产 `Cole_Hardware_Mug_Classic_Blue`，相对于基线 YCB `025_mug`，其杯身更宽、
更矮，整体为蓝色，把手环更宽且截面厚度不同。

摆放协议：

```text
cup world xy = (0.04, -0.10)
bowl world xy = (0.06, 0.10)
cup yaw       = -90 deg
handle        = camera image left
seed          = 0 for all stochastic stages
```

## 2. 一键复现

```bash
cd /home/users1/ljian/LFV

/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_cole_blue_mug_topdown.yaml
```

该配置包含 `snapshot_export`，会先构建新 mesh 资产并导出 RGB、depth、mask、
相机参数、完整视觉表面和场景点云，再运行二维迁移、完整表面传播、GraspNet 和
固定截图。替换新杯子时只需修改 `cup_visual_file`、`cup_collision_glob`、尺度、
颜色和摆放，不需要修改环境代码。

与原实例生成固定对比：

```bash
/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/compare_topdown_grasp_instances.py \
  --baseline-dir /home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/episode_0_to_maniskill_seed_0/topdown_grasp \
  --candidate-dir /home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/episode_0_to_cole_blue_mug_seed_0/topdown_grasp \
  --baseline-label "YCB 025 mug" \
  --candidate-label "Cole blue mug" \
  --output-dir /home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/episode_0_to_cole_blue_mug_seed_0/generalization_comparison
```

## 3. 结果

| 指标 | YCB 025 基线 | Cole 蓝杯 |
|---|---:|---:|
| 二维迁移 accepted | true | true |
| global confidence | 0.3366 | 0.4399 |
| cycle score | 0.0986 | 0.2015 |
| entropy | 0.6134 | 0.5775 |
| GraspNet decoded | 933 | 976 |
| 严格筛选后候选 | 3 | 4 |
| 选中抓取 final score | 0.5959 | 0.6407 |
| top-down angle | 6.3718° | 3.2909° |
| 左/右接触热力 | 0.8508 / 0.8850 | 0.8501 / 0.8493 |
| 接触对宽度 | 14.47 mm | 23.89 mm |
| 法向反平行程度 | 0.9728 | 0.9923 |
| 最大分部碰撞 IoU | 0.0 | 0.0 |

新杯二维热力峰值为 `(u=183,v=289)`。固定可视化显示峰值和主要热力质量均落在
左侧把手环，没有扩散到杯身或碗。完整表面阶段从 30000 个表面点中得到 768 个
反平行点对，其中 454 个连接不可见侧；最终接触点位于把手上部两侧，竖直高度差
为 1.37 mm。

因此，本实验对“同一源演示能够迁移到一个外观和几何都不同的带把手杯实例”
给出正面证据，而且新实例没有依赖放宽阈值。但是一个新资产不能证明总体同类别
泛化能力：当前新杯没有人工像素级把手 GT，热力定位依赖固定图像人工核验；抓取
也只通过静态点云碰撞筛选，尚未运行 Panda IK、夹爪闭合和提杯动力学。合理的
下一步是固定相同协议批量测试至少 10 个不同杯子资产并统计迁移接受率、把手定位
准确率、严格候选率和仿真执行成功率。

## 4. 固定产物

```text
/home/users1/ljian/lfv_runs/soft_heatmap_affcorrs/
└── episode_0_to_cole_blue_mug_seed_0/
    ├── transfer_result.npz
    ├── transfer_report.json
    ├── transfer_summary.png
    ├── topdown_grasp/
    │   ├── topdown_grasp_summary.png
    │   ├── topdown_grasp_pipeline_report.json
    │   ├── graspnet_selected.npy
    │   ├── graspnet_selected_world.npy
    │   ├── graspnet_selected_object.npy
    │   └── graspnet_selected_open3d.png
    └── generalization_comparison/
        ├── generalization_comparison.png
        └── generalization_comparison.json
```

快照单独保存在：

```text
/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/
└── cole_blue_mug_left_seed_0/
    ├── pouring_snapshot.npz
    ├── snapshot_report.json
    ├── rgb_base_camera.png
    └── cup_mask.png
```
