# LFV 文档索引

仓库只保留以下几份作为当前实现的入口文档；历史规划、阶段日志和重复实验记录已从活动文档中移除，删除内容仍可通过 Git 历史恢复。

## 推荐阅读顺序

1. [项目架构与开发约束](project_architecture_and_development_guide_zh.md)：目录、接口和快速迭代边界。
2. [完整方法材料](methods/LFV_complete_method_material_zh.md)：论文方法所需的完整计算流程和模块说明。
3. [Stage 1：AffCorrs + FGW Contact 迁移](stage1_affcorrs_fgw_contact_transfer_zh.md)：接触热力的来源、迁移算子和输出契约。
4. [Stage 2：当前方法](stage2/current_method_complete_zh.md)：Motion Field、Goal/Trajectory diffusion、训练和推理。
5. [严格相机推理](deployment/strict_camera_inference_zh.md)：RGB-D 到 `camera_plan.npz` 的正式入口。
6. [Aubo 执行交付](deployment/aubo_camera_execution_bundle_zh.md)：手眼变换、执行包和机器人端接入方法。

实验输出、checkpoint、数据集和运行录像不属于版本化文档，保留在各自的运行目录中。

