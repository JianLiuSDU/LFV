# Stage 2：稳定 Motion Functional Field 与跨实例融合改造计划

状态：计划，尚未修改代码。

本计划针对当前模型的两个理论缺口：第一，relevance head 目前只通过 Goal/Trajectory
denoising loss 间接学习，尚不能证明它对应稳定的任务功能区域；第二，FGW 迁移的
Motion Field prior 目前主要在推理时以固定权重混合，尚不能证明它在新杯子上真正
提升任务成功率。改造目标是在不堆叠大量人工特征的前提下，把 Motion Field 变成
一个可检验的关系瓶颈，并用多个未见过的杯子实例验证它对杯口位置和最终倒水目标的
实际作用。

## 1. 文献依据与 LFV 中的取舍

- [TAX-Pose](https://proceedings.mlr.press/v205/pan23a/pan23a.pdf) 用逐点重要性权重和
  可微加权对应求解任务相关的跨物体位姿；LFV 借鉴“重要性必须连接到任务损失”的
  思想，但不引入 TAX-Pose 的 SVD cross-pose 求解器，Goal/Trajectory 仍由扩散分支
  生成。
- [AffCorrs](https://proceedings.mlr.press/v205/hadjivelichkov23a/hadjivelichkov23a.pdf)
  使用冻结 DINO 稠密描述符和循环对应完成 one-shot 功能部件定位；LFV 继续将它用于
  Stage 1 Contact Field 的语义定位，不把 AffCorrs 输出冒充 Motion Field 标签。
- [DINOv2](https://arxiv.org/abs/2304.07193) 说明冻结自监督特征可以提供跨分布的
  稠密语义表示；它是跨杯子实例的语义证据，但不提供任务因果性，因此必须和结构
  对应及任务损失结合。
- [Fused Gromov--Wasserstein](https://proceedings.mlr.press/v97/titouan19a/titouan19a.pdf)
  同时优化节点特征和内部结构距离；LFV 用它把源 Motion Field 传输到目标完整杯子/
  碗点云。近期的 [Shape-of-You (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Im_Shape-of-You_Fused_Gromov-Wasserstein_Optimal_Transport_for_Semantic_Correspondence_in-the-Wild_CVPR_2026_paper.html)
  进一步说明 FGW 传输计划可以作为带噪软监督，但 LFV 第一版只采用其结构—语义
  对应启发，不引入大型新特征网络。

## 2. 改造后的核心定义

对操作物体和参考物体分别定义一个归一化的表面概率场：

\[
r^m_i = P(i\text{ 对当前任务关系有用}\mid X_m,X_r),
\qquad
r^r_j = P(j\text{ 对当前任务关系有用}\mid X_m,X_r),
\]

其中 \(X=(\mathrm{XYZ},\mathrm{DINO})\)，场的支撑是整个被分割物体，而不是仅有的
Contact 区域。这个定义不要求所有轨迹中的运动完全相同；要求的是在同一任务中，
不同初始位姿、速度和执行风格下，功能区域的概率质量相对稳定，而 Goal 和 Trajectory
可以变化。

改造后的因果链固定为：

```text
当前 XYZ-DINO evidence ─┐
                         ├─ field estimator ─┐
源实例 Motion Field ─FGW┘                    │
                                              ▼
                                  confidence-aware fusion
                                              ▼
                              field-weighted relational bottleneck
                                              ▼
                                  Goal diffusion → Trajectory diffusion
```

最终 decoder 只能读取 field-weighted 的三个关系 token，不再读取 unweighted global
feature 或完整逐点 relation token。这样可以真正检验“错误 Field 是否会破坏预测”，
而不是只展示一张漂亮热力图。

## 3. 网络改造（保持结构简洁）

### 3.1 Encoder 的硬瓶颈

保留当前两个 PointNet、双向 Cross-Attention 和 relevance head。修改点只包括：

1. `global token`、`manipulated-to-reference token` 和 `reference-to-manipulated token`
   全部由融合后的 (r^m,r^r) 加权聚合得到；
2. 禁止 `manipulated_global/reference_global → decoder` 的直接路径；
3. decoder 不接收未加权的逐点 relation features；
4. Field 的归一化、温度和有效支撑数量写入 checkpoint，便于跨实例比较；
5. 对场使用轻量的温度/熵约束，防止完全均匀或塌缩到单点，但不手工指定杯口坐标、
   PCA 轴或左右方向。

当前 `joint` 分支已经接近这个结构；需要补的是严格的接口约束和验证，而不是重新
设计一个更大的 Transformer。

### 3.2 让 Field 具有可检验的任务作用

在原有损失之外增加两个轻量约束：

\[
\mathcal L
=\mathcal L_{goal}+\mathcal L_{traj}
 +\lambda_{cons}\mathcal L_{field-cons}
 +\lambda_{cf}\mathcal L_{field-causal}.
\]

**同实例多轨迹一致性**：同一杯子不同 demonstration 或不同初始状态得到的场，经
过已知物体位姿/FGW 对齐后保持一致：

\[
\mathcal L_{field-cons}
 = D(r^{(a)},r^{(b)}).
\]

这里只利用已有轨迹分组和几何对应，不增加人工功能点标签。

**反事实 Field 约束**：对同一 noisy state 使用 learned、uniform、shuffled 或
complement field，要求 learned field 的 denoising loss 更低：

\[
\mathcal L_{field-causal}
 = \max(0,\,m+\mathcal L_{learned}-\mathcal L_{uniform/shuffled}).
\]

该项的作用是检验场是否承载了 Goal/Trajectory 所需信息，而不是直接把某个预设区域
当作监督。训练时先以较小权重启用，并和无该项的模型做消融。

## 4. Prior–Evidence Fusion 的改造

当前的算术混合：

\[
r_{fused}=(1-\alpha)r_{online}+\alpha r_{prior}
\]

保留为 baseline，但正式模型改为置信度感知的 log-opinion pool：

\[
r_{fused,i}
\propto
\exp\left((1-\alpha(x))\log(r_{online,i}+\epsilon)
             +\alpha(x)\log(r_{prior,i}+\epsilon)\right).
\]

其中 \(\alpha(x)\) 由一个很小的 gate 根据以下已有量预测：

- FGW cycle/transport confidence；
- prior 与 online field 的一致性；
- 两个场的归一化熵和有效支撑；
- 当前场景 DINO 特征的匹配质量。

gate 的输入是统计量和三个 scene token，不增加额外视觉 backbone。训练时随机使用：

```text
无 prior、正确 prior、部分遮挡 prior、打乱 prior、低置信度 prior
```

使网络学会在新实例上判断“相信源先验还是相信当前观测”。FGW prior 作为 stop-gradient
输入，避免把传输数值的噪声反向写入场估计器。

当 confidence 低于阈值时，不强行融合，退化为 evidence-only，并在报告中记录拒绝
原因。这样 Prior–Evidence Fusion 是一个可失败、可校准的模块，而不是固定的 0.5
超参数。

## 5. 数据和仿真验证设计

### 5.1 杯子实例

在 ManiSkill pouring 环境中建立至少四个带把手杯子实例，优先使用仓库已经支持的：

1. YCB `025_mug`；
2. `Cole_Hardware_Mug_Classic_Blue`；
3. 当前红色/ACE 杯资产；
4. 另一种带把手的 scanned mug 或 YCB mug。

如果某资产尺寸明显不同，先统一尺度范围并记录尺寸，不通过修改网络隐藏尺度差异。
每个杯子使用多个 yaw、平移和杯碗间距，至少保留一个完全未参与训练的杯子实例作为
cross-instance test。训练/验证/测试按 `cup_asset_id` 划分，不能只按帧随机切分。

### 5.2 源场和目标场

源 Motion Field 由训练集 checkpoint 导出为 memory。目标实例同时计算：

- `online field`：当前杯子和碗的 XYZ-DINO 经 encoder 得到；
- `transported prior`：源 memory 通过相同 FGW 算子传到目标完整点云；
- `fused field`：由 confidence gate 融合。

每个结果都保存：`online/prior/fused field`、FGW transport、gate weight、熵、峰值、
有效支撑和拒绝状态，并渲染到杯子点云和 RGB 图像上。

## 6. 正确的 pouring 成功标准：杯口，而不是杯身

仿真评估可以读取完整 mesh 的杯口 rim 点和碗口 opening 几何；这些几何只用于评估，
不输入网络。对最终对象位姿 \(T_g\)，将杯口采样点变换到碗坐标系，在碗口平面上
计算投影覆盖：

\[
\mathrm{ROF}
=\frac{1}{N_{rim}}
 \sum_{k=1}^{N_{rim}}
 \mathbf 1[\Pi(T_g p_k^{rim})\in\mathcal E_{bowl}]
 \cdot
 \mathbf 1[|z_k-z_{rim}|<\delta_z].
\]

ROF（rim-over-opening fraction）表示有多少杯口点位于碗口上方。主判据建议为：

```text
ROF ≥ 0.20
杯口至少有一段连续 rim arc 位于 opening 投影内
最终位姿和完整轨迹无碰撞
```

“有一部分杯口在碗上方”由 ROF 表达，不再要求整个杯身位于碗上方。为避免阈值
选择造成偏差，报告同时给出 ROF=0.10/0.20/0.30 的敏感性曲线。

同时记录：

- 杯口中心到碗口中心的平面距离；
- 杯口投影面积与碗口面积的重叠率；
- 杯口倾倒轴与碗口法向夹角；
- 轨迹中每一帧的 ROF、碰撞状态和末端速度/加速度；
- grasp 是否保持以及是否到达目标状态。

## 7. 必做对照和指标

### 7.1 Field 可解释性与因果性

对每个测试杯子报告：

- rim 区域 AUPRC/AUROC（仿真 rim 只作评估标签）；
- normalized entropy、peak mass、effective support；
- learned vs uniform/shuffled/complement field 的 Goal/Trajectory 损失差；
- Goal 平移误差、SO(3) 误差；
- Trajectory endpoint 误差、ROF 误差、碰撞率和任务成功率。

关键不是只看热力图，而是证明错误场会使杯口 ROF、终态误差或任务成功率显著恶化。

### 7.2 跨实例与融合

至少比较以下五组：

| 组别 | 当前 evidence | FGW prior | 融合方式 |
|---|---|---|---|
| A | 无 field/global baseline | 无 | 无 |
| B | online field | 无 | evidence-only |
| C | 无 | transported prior | prior-only |
| D | online + prior | 有 | 固定算术混合 |
| E | online + prior | 有 | confidence-gated log fusion |

再加入错误 prior、打乱 prior 和低置信度 prior，验证 gate 是否会降低 prior 权重或
拒绝迁移。每个杯子至少 5 个随机种子/初始摆放，使用 paired bootstrap 给出置信区间。

### 7.3 预注册验收门槛

建议在跑实验前固定以下门槛：

1. learned field 相比 uniform 的 rim AUPRC 和 ROF 误差必须有稳定改善，且 bootstrap
   置信区间不跨 0；
2. uniform/shuffled/complement 干预造成的最终 ROF 误差至少增加 10%，或任务成功率
   下降至少 10 个百分点，才可以声称 Field 具有任务必要性；
3. E 组在未见杯子上的 ROF/成功率优于 B、D，且不牺牲源杯子性能；
4. gate 权重与 FGW confidence 单调相关，错误 prior 的平均权重明显低于正确 prior；
5. 不能以“杯身位于碗上方”替代 ROF，也不能只报告单个成功视频。

这些数值是验收标准，不是训练目标；若数据规模不足以达到统计显著性，应报告效应
量和置信区间，而不是更改成功定义。

## 8. 代码落点和测试计划

预计只修改以下模块：

```text
lfv/models/functional_motion_generation/encoders/bidirectional_scene_encoder.py
    严格 field bottleneck、field consistency 统计、fusion gate
lfv/models/functional_motion_generation/motion_field_transfer.py
    prior confidence 和 transport diagnostics
lfv/models/functional_motion_generation/system.py
    causal/consistency loss 汇总
lfv/models/functional_motion_generation/trajectory/ 与 goal/
    保持原有 diffusion block 和接口不变
scripts/stage2/evaluate_motion_field.py
scripts/stage2/visualize_motion_fields.py
    多杯子、Field 干预和 ROF 指标
lfv_sim/maniskill/ 与 scripts/sim/
    仅增加杯口 rim/opening 的评估几何和多资产配置
configs/stage2/
    新增 cross-instance、fusion 和 ROF 阈值配置
tests/stage2/
    shape、bottleneck、fusion、错误 prior、ROF 判据和首帧一致性测试
```

不修改 Goal/Trajectory 的 9D 表示、DDPM 训练、DDIM 采样、64 步轨迹接口和 Aubo
交付格式。每一版改动都单独提交并打 tag，训练结果保存 best/last checkpoint、配置、
场可视化、干预报告和仿真录像。

## 9. 分阶段执行顺序

**P0：基线冻结。** 保存当前 joint checkpoint、当前 18-episode 测试结果和已有
`causality.json`；增加多杯子 snapshot 导出和 ROF 离线评估，不改网络。

**P1：硬瓶颈和场干预。** 固定 `joint` 模式，删除所有最终 decoder 的 unweighted
global 入口；完成 learned/uniform/shuffled/complement 的成对测试。

**P2：稳定场学习。** 加入同实例多轨迹 field consistency 和小权重反事实 ranking，
先在原 pouring 数据上 overfit/validation，再比较 Goal、Trajectory、ROF 和场熵。

**P3：Prior–Evidence Fusion。** 加入 prior dropout、错误 prior、confidence gate 和
log fusion；在留出的杯子实例上训练/验证，不把目标杯子轨迹泄漏到训练。

**P4：多杯子完整评估。** 对至少四个杯子、多个初始状态和多个随机种子运行 Stage 1
迁移、Stage 2 推理、top-down grasp 和完整轨迹；输出统一表格、场对比图、ROF 曲线、
碰撞检查和录像。

**P5：回归与论文证据。** 只有当 Field 干预、跨实例融合和杯口成功率同时通过门槛，
才把 Motion Functional Field 描述为稳定的任务先验；否则将论文表述限定为
“field-weighted relational conditioning”，并如实报告其作用范围。
