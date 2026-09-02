# LFV Stage 2 V7 分级验证报告

验证日期：2026-09-02
代码分支：`stage2/motion-functional-field`
验证时的最新代码：Gate-A 诊断脚本和结果生成脚本尚未提交，完成审核后单独提交。
本报告遵守 Gate A→B→C→D 的停止规则：前一 Gate 未通过时，不启动后续昂贵实验。

## 1. 数据和冻结边界

本轮只使用 Pouring 任务：

```text
cache: configured Pouring Stage-2 cache (`CACHE_ROOT`)
records: 179
train / val / test: 143 / 18 / 18
points per role: 256
DINO dimension: 384
```

V7 的结构、默认配置、损失、Field intervention 实现和 source-canonical memory
接口均保持冻结。只增加了两个诊断脚本和 V7 debug-only tensor 返回，未改变
generator 计算路径。旧 V2/V6 未修改。

需要特别注意：legacy cache 的 `object_instance_id` 为空，且没有可靠的
episode→canonical mapping。因此本轮不能声称严格跨实例泛化；正式配置的
`strict_instance_split: true` 会在读取到空 ID 时 fail-fast。

## 2. Gate A：计算图、旁路和梯度

### 2.1 实际 tensor shape

Pouring real-cache 固定 batch 的实际结果：

```text
manipulated_points / reference_points: [2,256,3]
manipulated_dino / reference_dino:     [2,256,384]
local E_m / E_r:                       [2,256,128]
selector feature_m / feature_r:        [2,256,128]
field_logits_m / field_logits_r:       [2,256]
gate_m / gate_r:                       [2,256]
relation R_m / R_r:                    [2,256,128]
functional_tokens_m / _r:              [2,4,128]
joint_token:                           [2,1,128]
Z_func:                                 [2,9,128]
goal prediction:                       [2,9]
trajectory prediction:                 [2,63,9]
```

训练/采样接口最终恢复为 64 帧轨迹，即 `[B,64,9]`；去噪 decoder 内部只预测
首帧之后的 63 帧，首帧由 hard-start boundary 拼回。

### 2.2 信息流验证

代码层面的边界为：

```text
E_m/E_r
   ↓
FieldSelector cross-attention
   ↓ 仅 field_logits/gate
GatedRelationEncoder(E_m,E_r, gate_m,gate_r)
   ↓
FunctionalPooling
   ↓
Z_func[9,128]
   ├── Goal Diffusion
   └── Trajectory Diffusion
```

Selector hidden state 不会传给 relation encoder、pooling 或任意 diffusion decoder。
诊断模式可以读取 selector feature，但这些张量只用于报告，不在生成器条件中使用。

### 2.3 Zero/uniform/shuffled 结果

real-cache 随机初始化模型、固定 timestep/noise 的结果如下：

| condition | context L2 | Goal pred L2 | Traj pred L2 | manipulated ratio | reference ratio | manipulated entropy |
|---|---:|---:|---:|---:|---:|---:|
| learned | 2.4791 | 1.9552 | 20.3994 | 0.4436 | 0.5272 | 0.99974 |
| uniform-budget | 2.4817 | 1.9665 | 20.4286 | 0.4436 | 0.5272 | 1.00000 |
| zero | 0.0000 | 2.4417 | 19.8365 | 0.0000 | 0.0000 | 0.00000 |
| rolled | 2.4804 | 1.9672 | 20.4177 | 0.4436 | 0.5272 | 0.99974 |
| shuffled | 2.4803 | 1.9651 | 20.4177 | 0.4436 | 0.5272 | 0.99974 |
| complement | 2.6297 | 1.9533 | 19.8185 | 0.5564 | 0.4728 | 0.99984 |
| bottom20 | 0.4700 | 2.0842 | 20.9880 | 0.0814 | 0.1033 | 0.70905 |

关键值：

```text
||Z_zero|| / (||Z_learned|| + eps) = 0.0
||Z_learned - Z_shuffled||_mean = 0.03211
Goal prediction delta (learned vs shuffled) = 0.02216
Trajectory prediction delta (learned vs shuffled) = 0.08533
```

这证明 Field 在实现上确实是可见的门控瓶颈；但此时 learned Field 几乎均匀，
rolled/shuffled 的影响很小，不能把“门控存在”解释成“已经学到了任务区域”。

### 2.4 仅任务损失的梯度

Field budget/smoothness/consistency 权重全部置零，只对任务损失反向传播：

| loss | Selector cross-attention | Field logits head | Local encoder | Gated relation | Pooling |
|---|---:|---:|---:|---:|---:|
| Goal only | 663.1620 | 177.3878 | 4184.2570 | 13427.4542 | 5959.3633 |
| Trajectory only | 43.7645 | 11.8546 | 285.0008 | 921.1857 | 418.1884 |

Goal/Trajectory 两个任务损失都能向 FieldSelector 传递非零梯度；这部分满足
“Field 不是只由正则项画出来”的要求。

### 2.5 固定 batch overfit

Synthetic 64 点、hidden=32 的 1000-step 固定 batch probe：

```text
total loss: 3.2016 → 0.0526
goal loss:  1.7473 → 0.0218
traj loss:  1.4540 → 0.0303
field entropy: 0.99875 → 0.99747
```

real pouring batch 的 100-step probe：

```text
total loss: 5.5587 → 0.3148
goal loss:  2.5535 → 0.1374
traj loss:  3.0051 → 0.1772
field entropy: 0.99990 → 0.99954
```

损失能够 overfit，但 Field 在 1000-step synthetic 和 real probe 中仍接近 uniform。
这不是结构旁路问题，而是当前 Field selector 尚未从运动监督中产生稀疏、稳定区域。

### 2.6 Gate A 判定

```text
selector hidden bypass absent:       PASS
zero-field context closed:            PASS
task-loss Field gradient non-zero:    PASS
Field changes context:                PASS（但 shuffled 差异较小）
fixed-batch loss decreases:           PASS
Field becomes non-uniform:            FAIL

Gate A: FAIL
```

停止原因：Gate A 要求 Field 在 overfit 后不能长期保持 uniform。当前 Field 的归一化
熵仍约 0.997–1.000，未满足该条件。因此本轮没有启动 Gate B 的正式 source training，
也没有启动 Gate C canonical stability 或 Gate D unseen-instance transfer。

## 3. Gate B/C/D 状态

```text
Gate B：NOT_RUN
原因：Gate A 未通过；未启动正式 3-seed source training。

Gate C：NOT_RUN
原因：Gate B 未通过；未聚合 canonical Field，也没有跨示范 consistency 结果。

Gate D：NOT_RUN
原因：Gate C 未通过；legacy cache object_instance_id 为空，且缺少可靠
      episode→canonical mapping，严格未见实例评估应 fail-fast。
```

## 4. 机器可读结果和图表

Gate A 报告（以下路径均相对于本地 `RUN_DIR`）：

- `RUN_DIR/gate_a_report.json`
- `RUN_DIR/gate_a_interventions.csv`
- `RUN_DIR/gate_a_interventions.png`
- `RUN_DIR/gate_status.json`
- `RUN_DIR/gate_status.csv`
- `RUN_DIR/gate_status.png`

诊断入口：

```bash
PYTHONPATH=. python scripts/stage2/validate_v7_gate_a.py \
  --config configs/stage2/motion_field_v7_pouring_lfv_smoke.yaml \
  --output-dir RUN_DIR --device cpu --batch-size 2 --overfit-steps 100

PYTHONPATH=. python scripts/stage2/summarize_v7_gate_status.py \
  --gate-a-report RUN_DIR/gate_a_report.json \
  --output-dir SUMMARY_DIR \
  --cache-root CACHE_ROOT
```

## 5. 结论边界

结论 1：Field 是否真正参与计算？

证据：Zero Field 时 `Z_func` 范数为 0；Field logits head 和 Selector 从 Goal/Trajectory
任务损失获得非零梯度；learned 与 shuffled 的 context 和预测存在差异。计算图层面
可以判定为 **是**。

结论 2：Field 是否在源实例上具有因果必要性和局部充分性？

证据：Zero/uniform/bottom intervention 会改变 context 和固定噪声预测；但 1000-step
probe 后 entropy 仍为 0.99747，尚未完成稀疏区域学习，也没有 deletion/insertion
性能曲线。因此只能判定为 **INCONCLUSIVE**，不能声称已证明局部充分性。

结论 3：Field 是否跨示范稳定？

证据：没有启动 Gate C；且当前缓存缺少可靠 canonical mapping。结论为
**NOT EVALUATED**。

结论 4：Source canonical alignment 是否改善严格未见实例泛化？

证据：没有启动 Gate D；`object_instance_id` 为空，严格 split 会 fail-fast。结论为
**NOT EVALUATED**。

结论 5：Field 主要影响 Goal、Trajectory，还是二者？

证据：Goal-only 和 Trajectory-only 都向 Selector/Field head 传递非零梯度；当前固定
噪声下 learned→shuffled 的 Goal L2 差异为 0.0222、Trajectory L2 差异为 0.0853。
这只能说明二者都可受 Field 影响，不能替代训练后误差和完整链路 intervention。

下一步：

1. 保持模型结构冻结，先定位为什么任务损失让 Field 保持 uniform；重点检查 gate
   temperature、budget curriculum、Field loss 权重和 DINO/XYZ 对任务区域的可分性，
   每项修改都建立独立配置和 commit。
2. 在 Gate A 通过前不启动 300 epoch、三 seed 或五任务训练。
3. Gate A 通过后才进行 V2/V7 公平 source 对照、deletion/insertion 和 paired bootstrap。
4. 重新生成带真实 `object_instance_id`、object-frame correspondence 和
   episode→canonical mapping 的 pouring 缓存后，才进入 Gate C/D。
