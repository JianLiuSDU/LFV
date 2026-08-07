# Stage 2 V2：双对象XYZ–DINO Encoder与三上下文Token设计

> 初稿：2026-08-05
> 第三次重审：2026-08-07
> 状态：最小网络架构设计，不包含代码修改。
> 相关文档：[当前计算流程](current_goal_and_trajectory_pipeline_zh.md)、[现有方法评价](research_review_and_architecture_assessment_zh.md)。

## 1. 本次简化后的结论

用户提出的认识基本正确。对于当前LFV Stage 2，在满足以下前提时，Encoder只输入两组`XYZ+DINO`，最终输出三个上下文特征，足以作为第一版结构：

1. manipulated与reference的mask已经由数据处理阶段给出；
2. pouring、drawer等任务分别训练独立checkpoint，不要求一个模型仅凭同一对象对区分多种任务；
3. Contact只服务Stage 1抓取区域和抓取生成，不负责Stage 2功能运动编码；
4. Stage 2学习的是object-centric task motion prior，不负责完整场景碰撞规划；
5. 碰撞、IK和机器人可达性由后级GraspNet、轨迹检查与执行模块处理。

在这些条件下，Stage 2 Encoder不需要：

- Contact heat；
- environment point cloud；
- language embedding；
- task type token；
- 复杂object/relation/environment token集合；
- 人工法向、曲率、协方差和KNN统计。

推荐的最小信息流为：

```text
manipulated XYZ + DINO              reference XYZ + DINO
            |                                  |
     Manipulated PointNet                Reference PointNet
            |                                  |
       Fm, gm                              Fr, gr
            |                                  |
            +------------+  +------------------+
                         |  |
                  bidirectional cross-attention
                         |  |
                         v  v
     z_initial      z_manipulated<-reference      z_reference<-manipulated
                         |
                         v
             context_tokens [B,3,C]
                  /                    \
                 v                      v
        Goal Pose Diffuser       Trajectory Diffuser
```

三个token分别回答：

1. 两个对象在初始状态下整体是什么样、相对布局是什么；
2. 对manipulated object而言，reference object的哪些区域和关系重要；
3. 对reference object而言，manipulated object的哪些区域和关系重要。

这比上一版`SceneEncoding`简单得多，也更适合作为可验证baseline。它的代价是后级不再显式获得局部三维anchor；如果实验发现精确旋转或局部几何泛化不足，再增加少量局部relation tokens，而不是第一版就引入。

## 2. Contact、环境、任务和语言为什么可以删除

### 2.1 Contact不进入Stage 2

当前Contact表示的是人手或机器人应该抓住/接触manipulated object的哪个区域，它对GraspNet和抓取位姿生成非常重要，但不必然决定功能运动目标：

- 杯子把手上的抓取区域不能直接告诉模型杯子最终应该位于碗口哪个位置；
- 抽屉把手中心的Contact不能直接告诉模型抽屉需要沿什么轨迹拉开；
- 功能运动关系主要来自manipulated/reference的语义、几何和轨迹监督。

因此Stage 1和Stage 2的统一不需要把Contact强行传入Stage 2。更清晰的接口是：

```text
共享的DINO语义基础
  ├─ Stage 1：DINO + source heat -> Contact迁移 -> grasp
  └─ Stage 2：DINO + XYZ -> goal/trajectory generation
```

这样两阶段共享DINO cache和像素—点映射，但各自只消费与自身目标相关的信号。

### 2.2 环境点云不进入Encoder

当前Stage 2训练标签来自manipulated object相对reference object的运动，而不是机器人全身无碰撞轨迹。若训练数据没有统一、可靠的桌面/机器人/障碍物点云监督，把environment tokens加入网络只会扩大输入分布。

删除环境输入后必须固定系统边界：

> Encoder和Trajectory Decoder只能学习对象间任务运动，不能声称生成结果天然无碰撞。

碰撞由生成后的full-scene collision check、IK与轨迹优化负责。

### 2.3 不输入task type和language的前提

如果一个checkpoint只训练pouring或只训练drawer，任务身份已经由dataset/config固定，language embedding和task token没有可辨识价值。删除它们更简洁。

如果未来同一对象对支持多个动作，例如“把杯子放到碗旁边”和“向碗中倾倒”，而两种动作在输入几何上完全相同，那么没有task/language条件的模型无法知道该执行哪一种。届时才需要恢复任务条件。

### 2.4 role不需要显式输入

manipulated与reference通过不同函数参数和不同PointNet分支进入网络，角色已经由结构编码，不需要再输出`object_role`或额外role token。

## 3. 最小数据接口

### 3.1 Encoder输入

```text
manipulated_points   Pm [B,Nm,3]
manipulated_dino     Dm [B,Nm,D]
reference_points     Pr [B,Nr,3]
reference_dino       Dr [B,Nr,D]
```

第一版固定$N_m=N_r=256$，只采样具有有效深度和有效DINO像素对应的可见RGB-D点，因此不需要padding mask或`dino_valid`进入网络。

如果后续使用补全点云，未观测背面没有真实DINO特征。第一版不要把补全点混进Stage 2 Encoder；完整点云可以继续供GraspNet和碰撞检查使用。

### 3.2 对齐约束

每个索引必须满足：

$$
(P_{m,i},D_{m,i}),\qquad(P_{r,j},D_{r,j})
$$

分别来自同一个RGB像素和深度反投影点。点云采样、DINO索引和随机置换必须同步执行。

DINO继续离线保存并冻结，checkpoint记录：

- DINOv2 backbone；
- feature layer与dimension；
- 图像crop、resize和padding方式；
- patch到原图坐标映射；
- DINO feature normalization。

### 3.3 公共坐标系

两个点集必须使用同一个局部坐标：

$$
\tilde P_m=(P_m-c_m)/s,
$$

$$
\tilde P_r=(P_r-c_m)/s.
$$

$c_m$是manipulated object质心，$s$是固定工作空间尺度或统一场景尺度。不能分别对两个对象独立去中心，否则会删除它们的相对平移。

`scene_origin`和`scene_scale`属于dataset/geometry adapter，不是Encoder输出token；输出位姿时由统一geometry模块恢复。

## 4. DINO投影与两个独立PointNet分支

### 4.1 共享DINO投影

DINO通常为768/1024维，不能未经投影直接与3维XYZ拼接。两个对象先使用同一个DINO projector：

$$
S_m=\operatorname{MLP}_{dino}(\operatorname{LN}(D_m))
\in\mathbb R^{B\times N_m\times64},
$$

$$
S_r=\operatorname{MLP}_{dino}(\operatorname{LN}(D_r))
\in\mathbb R^{B\times N_r\times64}.
$$

共享projector的原因是DINO本来处在一个公共语义空间，没有必要为两个角色学习两套不兼容投影。

建议：

```yaml
dino_projector:
  input_dim: D
  hidden_dim: 256
  output_dim: 64
  dropout: 0.1
```

### 4.2 XYZ投影

分别使用简单XYZ MLP：

$$
X_m=\operatorname{MLP}_{xyz}^{m}(\tilde P_m)
\in\mathbb R^{B\times N_m\times64},
$$

$$
X_r=\operatorname{MLP}_{xyz}^{r}(\tilde P_r)
\in\mathbb R^{B\times N_r\times64}.
$$

第一版不需要Fourier embedding、法向、曲率或额外距离统计。

### 4.3 低维融合后进入PointNet

先把两种64维特征拼接，再分别编码：

$$
U_m=[X_m,S_m]\in\mathbb R^{B\times N_m\times128},
$$

$$
U_r=[X_r,S_r]\in\mathbb R^{B\times N_r\times128}.
$$

两个PointNet架构相同但权重不共享：

$$
F_m=E_m(U_m)\in\mathbb R^{B\times N_m\times C},
$$

$$
F_r=E_r(U_r)\in\mathbb R^{B\times N_r\times C}.
$$

推荐$C=128$。每个PointNet只使用共享点MLP、LayerNorm/GELU和max pooling，不引入PointNet++层级采样：

```text
[XYZ projection 64, DINO projection 64]
                -> concat 128
                -> point MLP 128 -> 128 -> 128
                -> per-point F [B,N,128]
                -> max pool g [B,128]
```

得到：

$$
g_m=\max_iF_{m,i},\qquad
g_r=\max_jF_{r,j}.
$$

为什么PointNet分开：

- manipulated分支可以学习与“被移动对象”有关的特征；
- reference分支可以学习与“接收/约束对象”有关的特征；
- 两个分支都很小，完全分开不会像两套大PointNet++那样显著增加容量；
- 共享DINO projector仍保留共同语义基础。

## 5. 第一个特征：初始双对象上下文

初始环境特征直接合并两个全局PointNet特征：

$$
z_{init}=\operatorname{MLP}_{init}([g_m,g_r])
\in\mathbb R^{B\times C}.
$$

不额外拼接Contact、语言、任务类型、质心差、bbox和其他人工特征。因为两个PointNet都在同一个坐标系中处理XYZ，$g_m,g_r$已经能够编码对象形状、DINO语义和当前空间布局。

$z_{init}$表示：

- manipulated object整体语义和几何；
- reference object整体语义和几何；
- 两个对象当前初始配置的联合上下文。

## 6. 第二个特征：manipulated查询reference

### 6.1 Cross-attention

以manipulated逐点特征为query，reference逐点特征为key/value：

$$
A_{m\leftarrow r}
=\operatorname{softmax}\left(
\frac{(F_mW_q^m)(F_rW_k^r)^\top}{\sqrt{d_h}}
\right),
$$

$$
O_{m\leftarrow r}=A_{m\leftarrow r}(F_rW_v^r).
$$

直观上，每个manipulated点都在询问：

> 为了理解我未来应如何运动，reference object中哪些区域与我有关？

例如pouring中，杯口/杯体特征可能关注碗口和碗内区域；drawer中，把手特征可能关注抽屉箱体和滑动方向相关区域。

### 6.2 Relation feature

把query本身与取回的reference信息融合：

$$
R_{m\leftarrow r}
=\operatorname{FFN}_{mr}([F_m,O_{m\leftarrow r}])
\in\mathbb R^{B\times N_m\times C}.
$$

再进行全局max pooling：

$$
z_{m\leftarrow r}=\max_iR_{m\leftarrow r,i}
\in\mathbb R^{B\times C}.
$$

$z_{m\leftarrow r}$是一个方向明确的全局关系特征：以manipulated为观察主体，汇总它从reference中提取的任务相关信息。

### 6.3 如何判断reference什么区域重要

多头attention map：

$$
A_{m\leftarrow r}\in
\mathbb R^{B\times H\times N_m\times N_r}
$$

在query点和head上求平均：

$$
w_r(j)=\operatorname{mean}_{h,i}A_{m\leftarrow r}^{hij}.
$$

$w_r$可以直接映射回reference点云，显示“相对于manipulated而言，reference哪些区域重要”。它只作为调试和可视化输出，不进入正式`EncoderOutput`。

## 7. 第三个特征：reference查询manipulated

反方向使用独立参数：

$$
A_{r\leftarrow m}
=\operatorname{softmax}\left(
\frac{(F_r\bar W_q^r)(F_m\bar W_k^m)^\top}{\sqrt{d_h}}
\right),
$$

$$
O_{r\leftarrow m}=A_{r\leftarrow m}(F_m\bar W_v^m),
$$

$$
R_{r\leftarrow m}
=\operatorname{FFN}_{rm}([F_r,O_{r\leftarrow m}]),
$$

$$
z_{r\leftarrow m}=\max_jR_{r\leftarrow m,j}.
$$

这个方向回答：

> 对reference object而言，manipulated object的哪些区域、语义和几何决定了两者的功能关系？

对应的manipulated重要性可视化为：

$$
w_m(i)=\operatorname{mean}_{h,j}A_{r\leftarrow m}^{hji}.
$$

两个方向不能简单共享同一个attention层，因为query角色不同、需要学习的关系也不同。

## 8. 最终Encoder输出

### 8.1 唯一公共输出

把三个特征堆叠为：

$$
Z_{ctx}=\operatorname{stack}
([z_{init},z_{m\leftarrow r},z_{r\leftarrow m}],\dim=1)
\in\mathbb R^{B\times3\times C}.
$$

为了让后级明确区分三个token，加入三个固定learned type embedding：

$$
Z_{ctx}[:,k]\leftarrow Z_{ctx}[:,k]+e_k,
\qquad k\in\{init,mr,rm\}.
$$

这不是task type或language，只是说明当前token属于哪种结构角色。

正式接口缩减为：

```python
ContextEncoding(
    tokens,  # [B,3,128]
)
```

调试模式可以额外返回：

```python
debug = {
    "attention_manipulated_to_reference": A_mr,
    "attention_reference_to_manipulated": A_rm,
    "reference_importance": w_r,
    "manipulated_importance": w_m,
}
```

这些内容不写入Goal/Trajectory模型接口，也不进入checkpoint的normalizer。

### 8.2 为什么不再输出坐标、role和global relation

- `object_xyz`已经在PointNet输入中使用；
- role由两个分支和三个token位置编码；
- directional relation已经由两个cross-attention token表示；
- `global_relation`会与三个全局token重复；
- `scene_origin/scale`属于几何前后处理；
- environment和Contact已经从Stage 2职责中删除。

### 8.3 三个token是否真的够

作为第一版够，理由是：

1. 当前Goal只有9维、轨迹只有64×9维，不需要视觉生成模型级别的大token memory；
2. 当前旧GoalPose/Full64在更强全局压缩下已经能降低误差，说明任务不是必须保留每个点到最后；
3. 双向cross-attention在pooling前已经看过全部逐点XYZ–DINO特征；
4. 三个token职责固定，比几十个无监督token更容易调试；
5. 当前数据规模小，强结构压缩有助于控制过拟合。

但它不是理论上无损。max pooling后，局部对应只隐式存在于关系向量中，后级无法直接查询某个具体点。若出现以下现象，说明三个token不足：

- 新实例旋转明显错误；
- attention可视化正确，但Goal仍无法使用精确位置；
- 两个局部区域同时重要时被max pooling合并；
- 复杂绕障路径无法从全局关系恢复。

届时最小升级是每个方向保留4–8个relation tokens，而不是恢复上一版复杂`SceneEncoding`。

## 9. Encoder的置换性质

两个PointNet的point MLP对每点共享权重，max pooling对点顺序不敏感；cross-attention对query/key集合顺序保持等变，最终pooling再次得到不变全局特征。

因此随机置换两个点集不会改变三个context token的语义。点云不加入序列位置编码，XYZ本身就是空间输入。

## 10. 推荐Encoder网络配置

```yaml
scene_encoder:
  num_points_manipulated: 256
  num_points_reference: 256
  dino_dim: D
  dino_projected_dim: 64
  xyz_projected_dim: 64
  hidden_dim: 128
  point_mlp_layers: [128, 128, 128]
  cross_attention_heads: 4
  cross_attention_layers_each_direction: 1
  relation_ffn_dim: 256
  output_tokens: 3
  dropout: 0.1
```

第一版只使用一层每方向cross-attention。增加到两层应作为后续消融，不在初始实现中叠深。

## 11. 简化后的Goal Pose Decoder

Encoder不再输出局部anchor后，Goal Decoder也应同步简化，不再构造candidate-transformed object tokens或3D relative bias。

### 11.1 Goal状态

对刚体pouring仍输出：

$$
G=[t_x,t_y,t_z,r_1^\top,r_2^\top]\in\mathbb R^9.
$$

平移3D加rotation6D完整表示一个刚体$SE(3)$终态。低维输出不会限制条件特征维度；网络的条件是三个128维context token。

### 11.2 Goal token

对normalized noisy goal $G_k$和扩散时间$k$：

$$
q_G=\operatorname{MLP}_{goal}(G_k)
\in\mathbb R^{B\times1\times128}.
$$

三个Encoder token作为固定memory：

$$
M_G=Z_{ctx}\in\mathbb R^{B\times3\times128}.
$$

### 11.3 Goal diffusion block

每层只需要：

```text
1. timestep-conditioned AdaLN on goal token
2. cross-attention(query=goal token, key/value=3 context tokens)
3. residual FFN
```

推荐4层、4heads、hidden_dim=128。最后由goal token输出9D clean pose：

$$
\hat G_0=W_{out}q_G^{final}.
$$

这里确实能够运行扩散，因为被加噪随机变量是9维Goal，而三个context token提供条件：

$$
\tilde G_k=\sqrt{\bar\alpha_k}\tilde G_0
+\sqrt{1-\bar\alpha_k}\epsilon.
$$

推理时从不同$\epsilon$初始化得到多个Goal候选。

### 11.4 为什么删除candidate geometry

上一版显式把noisy Goal作用到局部object anchors，具有更强几何归纳偏置，但会重新引入object token、xyz和复杂接口。当前旧GoalPose已经证明“全局关系条件 + 9D pose denoising”可以训练，因此第一版先验证最小三token条件。

如果Goal在大旋转、新实例或精确局部关系上失败，candidate geometry是Goal Decoder的第一项升级，而不是Encoder第一版的默认组成。

## 12. Goal loss

平移归一化：

$$
\tilde G=[(t-\mu_t)/\sigma_t,r_{6D}],
$$

其中$\mu_t,\sigma_t$只由训练集计算，rotation6D保持标准表示并在输出时正交化。

网络第一版沿用当前已验证的clean-sample prediction：

$$
\mathcal L_{diff}^{G}
=\frac{1}{9}\|\hat{\tilde G}_0-\tilde G_0\|_2^2.
$$

反归一化并把rotation6D转为矩阵后：

$$
\mathcal L_t=\operatorname{SmoothL1}(\hat t,t),
$$

$$
\mathcal L_R=
\arccos\left(
\operatorname{clamp}
\frac{\operatorname{tr}(\hat R^\top R)-1}{2},
-1+\epsilon,1-\epsilon
\right).
$$

总损失：

$$
\mathcal L_G=
\mathcal L_{diff}^{G}
+\lambda_t\mathcal L_t
+\lambda_R\mathcal L_R.
$$

最小版本不再加入anchor loss，因为Encoder没有向Goal Decoder公开anchors。建议初始权重：

```yaml
goal_loss:
  diffusion: 1.0
  translation: 1.0
  rotation_geodesic: 0.5
```

## 13. 多Goal的使用方式

Goal Decoder推理时生成$K_g$个候选：

```text
context [B,3,C]
  -> repeat to [B*Kg,3,C]
  -> Kg independent Gaussian initial states
  -> Goal diffusion
  -> goals [B,Kg,9]
```

多个Goal不能无标识地共同条件一条轨迹。每个Goal独立生成对应轨迹：

```text
goal 0 -> trajectory 0_0, 0_1
goal 1 -> trajectory 1_0, 1_1
...
```

对同一个GT添加多个小噪声只能提高局部鲁棒性，不能制造数据中不存在的真实多模态。多模态仍来自不同演示终态、对象对称性和不同扩散初始噪声。

## 14. 简化后的Trajectory Decoder条件

对每个Goal候选$G^{(k)}$，生成一个Goal condition token：

$$
z_G^{(k)}=\operatorname{MLP}_{goal\_cond}(G^{(k)})
\in\mathbb R^{B\times1\times128}.
$$

Trajectory Decoder的全部context memory为：

$$
M_\tau=[Z_{ctx},z_G]
\in\mathbb R^{B\times4\times128}.
$$

不再输入：

- Contact；
- environment tokens；
- language/task type；
- object/relation xyz；
- transformed goal geometry tokens。

这四个memory token分别表示初始双对象状态、两个方向关系和当前Goal候选。

## 15. Trajectory Diffusion Transformer

### 15.1 轨迹状态

pouring继续使用：

```text
trajectory [B,64,9]
pose = translation3 + rotation6d
```

轨迹中第$i$个noisy pose转成token：

$$
y_i=\operatorname{MLP}_{traj}(\tau_{k,i})+e_{progress}(i/63).
$$

### 15.2 Trajectory block与Goal block不同

每层：

```text
1. diffusion-timestep AdaLN
2. local temporal Conv1D, kernel=3
3. bidirectional temporal self-attention over 64 trajectory tokens
4. cross-attention(query=trajectory, key/value=4 context/goal tokens)
5. residual FFN
```

区别在于：

- Goal Decoder只有一个随机位姿，核心是从三个场景token读取条件；
- Trajectory Decoder有64个有序状态，必须同时学习局部连续性和长程顺序；
- temporal Conv1D提供局部平滑归纳偏置；
- self-attention表达“先移动、后旋转”等全局时序；
- cross-attention让不同时间步根据需要读取初始关系和Goal。

整个轨迹一次性双向去噪，不使用causal mask。

### 15.3 Soft endpoint

只硬固定起点：

$$
\tau_{k,0}=I.
$$

第1到63帧全部加入噪声并由模型生成，最后一帧不再强制等于输入Goal：

$$
G^{refined}=\hat\tau_{0,63}.
$$

训练Trajectory时对GT Goal加入与Goal Decoder验证误差匹配的小幅SE(3)扰动：

```text
1份 clean GT Goal
2份 small perturbed GT Goal
可选1份经过near-GT阈值筛选的Goal Decoder sample
```

模型学习在Goal附近修正终态并生成连续路径。明显错误的Goal不作为正轨迹条件，应由候选筛选拒绝。

## 16. Trajectory loss

起点不参与diffusion loss，第1到63帧计算：

$$
\mathcal L_{diff}^{\tau}
=\frac{1}{63}\sum_{i=1}^{63}
w_i\|\hat{\tilde\tau}_{0,i}-\tilde\tau_{0,i}\|_2^2,
$$

其中$w_{63}=2$，其余为1，使自由终点得到更强监督。

物理位姿损失：

$$
\mathcal L_{trans}^{\tau}
=\frac{1}{63}\sum_{i=1}^{63}
\operatorname{SmoothL1}(\hat t_i,t_i),
$$

$$
\mathcal L_{rot}^{\tau}
=\frac{1}{63}\sum_{i=1}^{63}
d_{SO(3)}(\hat R_i,R_i).
$$

局部速度损失：

$$
\mathcal L_{vel}
=\frac{1}{63}\sum_{i=0}^{62}
\| (\hat t_{i+1}-\hat t_i)-(t_{i+1}-t_i)\|_1.
$$

终点监督：

$$
\mathcal L_{end}
=\|\hat t_{63}-t_G\|_1
+\lambda_{end,R}d_{SO(3)}(\hat R_{63},R_G).
$$

总损失：

$$
\mathcal L_\tau
=\mathcal L_{diff}^{\tau}
+\lambda_t\mathcal L_{trans}^{\tau}
+\lambda_R\mathcal L_{rot}^{\tau}
+\lambda_v\mathcal L_{vel}
+\lambda_e\mathcal L_{end}.
$$

## 17. 推荐完整配置

```yaml
model:
  context_dim: 128

  encoder:
    point_count: 256
    dino_projected_dim: 64
    xyz_projected_dim: 64
    point_hidden_dim: 128
    separate_pointnet_weights: true
    cross_attention_heads: 4
    cross_attention_layers: 1
    output_tokens: 3

  goal_decoder:
    pose_dim: 9
    hidden_dim: 128
    num_layers: 4
    num_heads: 4
    train_diffusion_steps: 100
    inference_steps: 20
    prediction_type: sample

  trajectory_decoder:
    horizon: 64
    pose_dim: 9
    hidden_dim: 128
    num_layers: 6
    num_heads: 4
    temporal_conv_kernel: 3
    train_diffusion_steps: 100
    inference_steps: 20
    hard_condition_start: true
    hard_condition_goal: false
```

pouring使用9D rigid pose adapter。drawer若继续单独训练，可在配置中替换为prismatic state adapter；这属于输出状态配置，不需要把`task_type`送入Encoder。

## 18. 训练流程

### 18.1 Encoder + Goal

```text
Pm,Dm -> PointNet_m -> Fm,gm
Pr,Dr -> PointNet_r -> Fr,gr

gm,gr                    -> z_initial
Fm query Fr              -> z_m<-r
Fr query Fm              -> z_r<-m

stack -> context [B,3,128]
context + noisy goal -> Goal Decoder -> L_goal
```

### 18.2 Trajectory

```text
context [B,3,128]
clean/perturbed goal -> goal token
noisy trajectory -> 64 trajectory tokens

temporal self-attention
cross-attention to [context,goal]
-> full trajectory + refined endpoint
```

### 18.3 联合微调

先训练Encoder+Goal，再训练Trajectory，最后使用小学习率联合微调。Goal与Trajectory分别采样独立扩散时间步。

## 19. 推理流程

```text
1. 从同一RGB-D帧提取两组可见点云和逐点DINO
2. Encoder计算一次context_tokens [B,3,128]
3. Goal Decoder从不同初始噪声生成Kg个9D Goal
4. 对每个Goal单独生成Kt条trajectory
5. 每条trajectory返回refined endpoint
6. 按Goal关系、修正幅度、平滑度进行排序
7. 后级使用完整点云进行碰撞、IK和执行检查
```

推荐$K_g=8,K_t=2$。

## 20. 最小接口

```python
context = encoder(
    manipulated_points=batch["manipulated_points"],
    manipulated_dino=batch["manipulated_dino"],
    reference_points=batch["reference_points"],
    reference_dino=batch["reference_dino"],
)
# context.tokens: [B,3,128]

goal_losses = goal_decoder.compute_loss(
    context=context.tokens,
    goal_state=batch["goal_state"],
)

goals = goal_decoder.sample(
    context=context.tokens,
    num_samples=K_goal,
)

trajectory_losses = trajectory_decoder.compute_loss(
    context=context.tokens,
    goal_conditions=training_goal_conditions,
    trajectory=batch["trajectory_state"],
)

trajectories = trajectory_decoder.sample(
    context=context.tokens,
    goals=goals,
    num_samples_per_goal=K_trajectory,
)
```

## 21. 必须保存的Encoder可视化

虽然正式输出只有三个token，双向attention map必须作为固定调试基础设施保存：

```text
input_manipulated_dino_projection.png
input_reference_dino_projection.png
reference_importance_from_manipulated.png
manipulated_importance_from_reference.png
cross_attention_matrix_mr.png
cross_attention_matrix_rm.png
```

对点云使用Open3D颜色显示$w_r,w_m$，可以直接判断：

- 模型是否关注碗口而不是碗底/背景；
- 模型是否关注杯口、杯体或把手；
- drawer是否关注黑色把手和箱体对应区域；
- attention是否均匀塌缩或只关注单个异常点。

注意：attention是网络内部权重，不自动等于真实物理对应。它用于诊断，不能单独作为模型正确性的证据。

## 22. 第一版消融

### 22.1 Encoder输入

1. XYZ only；
2. DINO only；
3. XYZ + DINO。

用于验证DINO是否提供跨实例语义、XYZ是否提供必要空间信息。

### 22.2 Encoder结构

1. 两个PointNet全局特征直接concat；
2. concat + 单向cross-attention；
3. concat + 双向cross-attention；
4. PointNet权重共享 vs 不共享；
5. 三token输出 vs 三token再压成一个token。

### 22.3 Goal与Trajectory

1. 当前GoalPose/Full64；
2. 三context token + Goal diffusion；
3. hard endpoint vs soft endpoint；
4. clean Goal vs perturbed Goal；
5. GT Goal vs Goal Decoder samples。

## 23. 对当前认识的最终判断

你的三个特征设计是合理的，而且比上一版更适合作为第一版：

```text
初始双对象特征
manipulated <- reference关系特征
reference <- manipulated关系特征
```

它保留了最重要的三类信息：单对象语义/几何、双对象初始布局、双向功能关系，同时删除了当前没有可靠监督或不属于Stage 2职责的Contact、环境、任务、语言和复杂局部token。

需要保留的判断边界是：三个token是**强压缩的最小上下文**，适合验证网络是否能学会当前pouring/drawer任务，但不能提前宣称它对所有复杂局部关系都足够。如果attention可视化已经找对区域、而Goal仍无法精确落到目标位置，说明问题不是继续增加DINO或Contact，而是三token压缩丢失局部几何；届时只需增加少量局部relation tokens。

## 24. 最终结构

```text
Pm XYZ + DINO                         Pr XYZ + DINO
      |                                      |
PointNet_m                              PointNet_r
      |                                      |
    Fm,gm                                  Fr,gr
      |                                      |
      |------ manipulated queries reference--|
      |------ reference queries manipulated--|
      |                                      |
      +---------------+----------------------+
                      |
        [z_init, z_m<-r, z_r<-m]
                 [B,3,128]
              /             \
             v               v
      9D Goal Diffuser    Trajectory DiT
             |               |
        K Goal samples       |
             +-------+-------+
                     v
       64-step trajectory + refined endpoint
                     |
              collision / IK / execution
```

这个版本的Encoder只有两个PointNet、两个方向各一层cross-attention和三个输出token，结构足够简洁，也完整响应了“XYZ+DINO、双对象分别编码、双向关系交互、三个上下文特征”的设计目标。
