# 从二维迁移热力到仿真完整点云 Top-down GraspNet 抓取

更新日期：2026-07-31

## 1. 目标与边界

这一阶段消费已经稳定的二维 Soft Heatmap AffCorrs 输出，并在 ManiSkill 提供的
完整杯子几何上生成一个满足以下条件的抓取：

1. 两个夹爪接触侧都位于任务热力区域；
2. 两个接触点是完整杯面上的反平行表面对，能够跨过杯把手两侧；
3. approach 尽量接近世界坐标系 `-Z`，允许小角度倾斜；
4. 杯子完整表面、桌面、碗和场景点均参与碰撞检查；
5. 保存相机视角、完整热力点云和 Open3D 夹爪截图。

这里的“完整点云”不是从单视角图像学习补全出来的。ManiSkill 已知杯子的碰撞
mesh，本阶段从该 mesh 均匀采样 30000 个带外法向的完整表面点。二维热力只负责
决定任务区域，完整仿真几何负责补足不可见表面和验证夹爪可行性。

当前只做静态抓取位姿生成和点云碰撞检测，还没有做 Panda IK、闭合仿真、提杯
或 pouring rollout。因此报告中的“无碰撞”准确含义是：在当前 8 mm voxel
模型无关碰撞检测器中，最终候选的左右手指、掌部和 approach path 占用均通过
严格门限；它不是动力学成功率的替代。

## 2. 一键运行

```bash
cd /home/users1/ljian/LFV

/home/users1/ljian/anaconda3/envs/tapip3d/bin/python \
  scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_maniskill_topdown.yaml
```

只调整后半段参数时可以复用已有结果：

```bash
.../tapip3d/bin/python scripts/sim/run_transferred_heat_topdown_grasp.py \
  --config configs/affordance_grasp/episode0_to_maniskill_topdown.yaml \
  --skip-transfer --skip-lifting
```

流水线分别使用：

```text
tapip3d env  : DINOv2、二维迁移、RGB-D lifting、报告合成
graspnet env : GraspNet、collision detector、Open3D/Xvfb
```

每个阶段的完整 stdout/stderr 保存在 `topdown_grasp/pipeline_logs/`。

需要验证新杯子实例时，使用
`configs/affordance_grasp/episode0_to_cole_blue_mug_topdown.yaml`。该配置会额外
执行 stage 00 导出自定义 mesh 快照；具体受控对比见
`docs/cole_blue_mug_generalization_validation_zh.md`。

## 3. 输入和坐标系

### 3.1 二维热力

```text
transfer_result.npz["target_heatmap"]
shape: [480,640]
range: [0,1]
frame: ManiSkill base_camera RGB pixel frame
```

### 3.2 仿真快照

```text
rgb                    [480,640,3]
depth_m                [480,640]
cup_mask               [480,640]
intrinsic_cv           [3,3]
full_points_camera     [30000,3]
full_normals_camera    [30000,3]
scene_points_camera    [M,3]
T_world_to_camera      [4,4]
T_object_to_camera     [4,4]
```

所有 GraspNet 候选先在 OpenCV camera frame 中计算。最终同时保存 camera、world
和 mug object 三个坐标系的 GraspNet 17 维 row。

GraspNet row 的有效字段为：

```text
score[1], width[1], height[1], depth[1], rotation[9], translation[3], object_id[1]
```

旋转矩阵第一列是 approach，第二列是 closing axis。

## 4. 二维热力提升到可见三维表面

对杯子 mask 内具有合法深度的每个像素：

\[
z=D(u,v),\quad
x=(u-c_x)z/f_x,\quad
y=(v-c_y)z/f_y.
\]

得到：

```text
visible_pixels_uv      [Nv,2]
visible_points_camera  [Nv,3]
visible_heat           [Nv]
```

当前固定案例有 5591 个有效杯子深度点。二维热力阈值为 `0.15`，424 个点保留
非零任务热力。阈值以下的点仍作为“已被相机看见的几何”保留，只把热力置零，
这样不可见面判断不会把已观察的冷区域误当成隐藏区域。

实现：`lfv/lifting/image_heat_to_surface.py`。

## 5. 从可见热力到完整表面反平行接触场

### 5.1 投到完整表面

对每个可见深度点，用 KD-tree 找最近完整 mesh sample；距离不超过 6 mm 时接受。
在同法向局部邻域内用高斯核扩散，得到 `projected_visible_heat`。

### 5.2 寻找把手另一侧

对于高热完整表面点 `p`、法向 `n_p` 和候选另一侧点 `q`：

\[
c=\frac{q-p}{\|q-p\|}.
\]

要求：

```text
4 mm <= ||q-p|| <= 30 mm
-c dot n_p       >= 0.55
 c dot n_q       >= 0.55
-n_p dot n_q     >= 0.55
```

这三个条件分别保证可见侧法向、另一侧法向和两表面法向相互满足平行夹爪的
反平行关系。只有属于有效反平行点对的区域进入最终 `full_heat`，而不是按欧氏
距离直接穿过杯壁扩散。

固定案例输出：

```text
完整表面点                         30000
投影成功的可见点                    5438
反平行点对                           768
其中连接到不可见表面的点对           261
点对宽度中位数                     13.26 mm
```

保存为 `transferred_contact_3d.npz`。

## 6. GraspNet 候选生成

输入采样 25600 点：

```text
25% 从 full_heat >= 0.10 的完整杯面按热力加权采样
75% 从热区邻近杯面和场景背景采样
```

不把小热区作为 GraspNet 的硬 workspace mask，否则网络会失去杯身、桌面和碗的
几何上下文。热区在解码后用于点对细化和硬筛选。

当前运行：

```text
GraspNet decoded                         933
保留用于点对细化                         300
生成的反平行细化候选                    5609
进入碰撞矩阵的候选                      1200
通过标准 GraspNet collision flag         564
通过热区/top-down 条件                      5
通过严格分部碰撞条件                        3
```

## 7. Top-down 与跨把手细化

世界期望 approach 为：

```text
d_world = [0,0,-1]
d_camera = R_world_to_camera d_world
```

每个 GraspNet proposal 只向邻近的有效反平行点对展开。点对 chord 作为 closing
axis；期望向下方向先投影到 chord 的正交平面，构成合法 approach：

\[
a=\mathrm{normalize}(d-c(c^Td)),\qquad
b=a\times c.
\]

最终旋转为 `[a,c,b]`。夹爪中心沿 approach 后退 depth，使两指尖落到点对中心；
开口为：

```text
gripper_width = pair_width + 10 mm clearance
```

硬约束：

```text
approach 与世界 -Z 夹角              <= 25 deg
接触对沿世界竖直方向高度差           <= 5 mm
左右指尖热力各自                      >= 0.20
指尖到完整表面平均距离                <= 7 mm
```

## 8. 严格碰撞筛选

碰撞点包含完整杯子点云和当前 RGB-D 场景中所有非杯点。先使用 GraspNet 官方
`ModelFreeCollisionDetector`，再对候选的每个部件单独加门限：

```text
global IoU         <= 0.02
left finger IoU    <= 0.02
right finger IoU   <= 0.02
palm IoU           <= 0.01
approach path IoU  <= 0.01
```

严格筛选前的候选及每个部件的碰撞量始终保存到：

```text
graspnet_ranked_before_strict_collision.npy
graspnet_strict_collision_diagnostics.json
```

如果没有候选通过，流水线直接失败，不自动放宽门限。

## 9. 当前最终抓取

```text
approach 到世界 -Z 夹角             6.3718 deg
GraspNet/refine final score          0.5959
左右指尖热力                         0.8508 / 0.8850
接触对宽度                           14.473 mm
夹爪开口                             24.473 mm
接触对高度差                         1.606 mm
指尖平均表面距离                     4.951 mm
可见侧法向对齐                       0.9831
另一侧法向对齐                       0.9914
两侧法向反向程度                     0.9728
global/left/right/palm/path IoU      全部 0.0
```

抓取位姿：

```text
camera translation = [-0.137707,  0.087549, 0.567093]
world translation  = [ 0.089277, -0.137707, 0.083751]
object translation = [ 0.049277, -0.037707, 0.035023]
```

## 10. 固定产物

```text
topdown_grasp/
├── transferred_contact_3d.npz
├── transferred_contact_3d_report.json
├── transferred_heat_lift_overlay.png
├── contact_full_camera_view.png
├── contact_full_camera_closeup.png
├── graspnet_selected.npy
├── graspnet_selected_world.npy
├── graspnet_selected_object.npy
├── graspnet_full_candidates_ranked.npy
├── graspnet_full_contact_report.json
├── graspnet_strict_collision_diagnostics.json
├── graspnet_selected_rgb_clean.png
├── graspnet_selected_open3d.png
├── topdown_grasp_summary.png
├── topdown_grasp_pipeline_report.json
└── pipeline_logs/
```

`topdown_grasp_summary.png` 固定显示：二维热力、完整表面热力、仿真 RGB 抓取和
Open3D 完整热力/夹爪，并在标题中显示倾角、点对宽度、两侧热力和最大碰撞量。

## 11. 测试与快速迭代约束

新增测试覆盖：

- RGB/heat/depth/intrinsic 的逐像素反投影对齐；
- heat threshold 不改变可见几何集合；
- 严格碰撞 mask 同时检查五个夹爪部件；
- 固定四联抓取图生成。

调参时优先看：

1. `transferred_heat_lift_overlay.png` 是否仍只在把手；
2. `contact_full_camera_closeup.png` 是否把热力扩展到把手两侧；
3. strict collision diagnostics 是手指、掌部还是 approach path 失败；
4. selected 的双指热力、接触对几何和 top-down angle；
5. 最后再看总 score，不能用高 GraspNet score 覆盖硬约束失败。
