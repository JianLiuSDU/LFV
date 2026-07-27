# LFV

LFV is now a compact codebase for learning robot manipulation from human
RGB-D videos.

The repository keeps:

- reusable RGB-D data processing code;
- contact heat and thumb-index grasp label generation;
- third-party wrappers and local third-party source links;
- the skeleton for the new two-stage generation framework.

Large datasets and generated outputs are not stored in this repository. They
live under `/media/ljian/lj`.

## Current Focus

The new model design is a two-stage multimodal generation framework:

1. `Contact-Grasp Generation Network`
   - input: manipulated object point cloud, normals, DINO point features;
   - output: task contact heat field and robot gripper grasp candidates.

2. `Functional Motion Generation Network`
   - input: manipulated object, reference object, DINO/geometric relation;
   - output: object-centric SE(3) functional motion trajectories.

Robot feasibility selection will combine sampled grasps and sampled motion
trajectories using IK, smoothness, joint limits, and collision checks.

## Important Docs

- `docs/hand_pouring_contact_and_grasp_handoff.md`
  - current data-labeling status and episode_0 outputs.
- `docs/project_architecture_and_development_guide_zh.md`
  - Chinese guide for project structure, model development, training,
    inference, robot feasibility checks, and simulation evaluation.
- `docs/repository_reorganization_for_two_stage_generation.md`
  - current project structure, cleanup record, and future development rules.
- `docs/hamer_grasp_pseudo_label_plan.md`
  - HaMeR thumb-index grasp pseudo-label design.

## Data Processing

Main pipeline entry:

```bash
python scripts/run_pipeline.py --config configs/pipeline/hand_pouring.yaml --steps prepare dino sam2 sample hand_bbox hand_mask timing contact_heatmap dinov2 hamer thumb_index_grasp
```

Batch helpers:

```bash
bash scripts/run_hand_pouring_contact_batch.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_hand_pouring_grasp_batch.sh
```

Processed hand-pouring data:

```text
/media/ljian/lj/data_3d/hand_pouring_lfv
```

## New Architecture Skeleton

Code packages:

```text
lfv/data_processing/              preprocessing API and episode I/O helpers
lfv/datasets/                     future model datasets
lfv/models/contact_grasp_generation/
lfv/models/functional_motion_generation/
lfv/training/
lfv/inference/
lfv/evaluation/
lfv/robot/
lfv/visualization/
```

Configs:

```text
configs/data/
configs/experiments/
configs/pipeline/
```

Scripts:

```text
scripts/preprocess/
scripts/train/
scripts/infer/
scripts/evaluate/
scripts/visualize/
scripts/robot/
```

## Third-Party Code

Third-party code stays under `third_party/`:

```text
third_party/dinov2_weights
third_party/hamer -> /home/users1/ljian/hamer
third_party/sam2
third_party/tapip3d
```

HaMeR should be launched through:

```bash
bash scripts/run_hamer_demo_env.sh ...
```

This uses `/home/users1/ljian/anaconda3/envs/hamer/bin/python`.

## Legacy Backup

Before cleanup, a lightweight backup excluding `data/`, `third_party/`,
`.git`, and caches was created at:

```text
/home/users1/ljian/LFV_legacy_20260727_no_data_no_third_party
```

The current repository no longer keeps old training/model/simulation code.
Those parts should be rewritten under the new structure.
