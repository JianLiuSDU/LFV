# Motion Field P0/P1 公共评估摘要

本文件只记录可公开复现的评估协议和汇总数值，不包含服务器、数据集或缓存的绝对路径。
完整运行命令中的路径应由使用者在本地替换为自己的 `RUN_ROOT` 和 `CACHE_ROOT`。

## 配对评估协议

`evaluate_motion_field_matrix.py` 对每个 episode、每个 checkpoint 和每个 Field 条件
使用相同的 episode seed，并在 episode 级别进行 bootstrap。v6 结果使用 18 个 test
episode、3 个随机种子、每个输入 4 个 Goal sample 和 1 条 Trajectory sample。

Field 条件包括：

- learned、uniform、roll、complement；
- keep-top 5/10/20%（insertion）；
- drop-top 5/10/20/30%（deletion）。

## v6 Goal 结果

| 条件 | Goal top-1 平移误差 | 相对 learned |
|---|---:|---:|
| learned | 51.68 mm | 0 |
| uniform | 54.08 mm | +2.40 mm [0.97, 3.93] |
| roll | 53.92 mm | +2.24 mm [0.87, 3.62] |
| complement | 55.29 mm | +3.60 mm [1.55, 5.75] |
| drop-top 5% | 52.57 mm | +0.89 mm [0.41, 1.38] |
| drop-top 10% | 53.06 mm | +1.38 mm [0.54, 2.27] |
| drop-top 20% | 53.95 mm | +2.27 mm [0.82, 3.80] |
| drop-top 30% | 54.90 mm | +3.22 mm [1.12, 5.34] |
| keep-top 5% | 47.71 mm | -3.98 mm [-7.87, 0.15] |

删除比例增加时误差单调增加，说明 Field 的高质量区域承载了 Goal 信息；但只保留
少量高质量区域反而更好，说明当前 Field 尾部仍包含噪声。Trajectory 的对应误差约
71 mm，删除 Field 后没有稳定退化，因此目前只能声称 Field 对 Goal 有可测因果作用。

## 当前数据限制

当前 pouring 缓存的 `object_instance_id` 全为空，train/val/test 是 episode split，
不是 object-instance split。因此本结果不能作为严格的跨实例泛化证据。后续必须先恢复
实例 ID，再使用未见杯子资产进行 transfer/fusion 验收。

## 结论

下一轮应优先校准 Field 的稀疏性、修复实例划分并加入杯口 ROF、碰撞率和任务成功率。
不应仅凭 denoising loss 或热力图视觉效果声称 Motion Field 已经提升了实际任务成功率。

