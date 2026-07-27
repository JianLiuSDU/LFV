# Repository Cleanup And Two-Stage Generation Structure

## Summary

This repository has been cleaned and reorganized for the next phase of LFV:
learning robot manipulation from human RGB-D videos with a two-stage multimodal
generation framework.

The cleanup was intentionally bold:

- old generated `data/` outputs were removed from the repository;
- old model/training code was removed;
- old simulation code was removed;
- old one-off step wrappers were removed;
- third-party dependencies were kept;
- proven data-processing code was kept;
- key visualization and validation tools were kept;
- a clean package skeleton for the new models was added.

Before cleanup, a lightweight backup was created at:

```text
/home/users1/ljian/LFV_legacy_20260727_no_data_no_third_party
```

That backup excludes `data/`, `third_party/`, `.git`, caches, and bytecode.

## What Remains

Top-level retained structure:

```text
configs/
docs/
legacy/
lfv/
scripts/
tests/
third_party/
tools/
README.md
pyproject.toml
requirements.txt
```

The repository size after cleanup is about 250MB, dominated by `third_party/`.

## Retained Data Processing Code

The current working data pipeline remains in:

```text
lfv/pipeline/
scripts/run_pipeline.py
```

Important pipeline modules:

```text
lfv/pipeline/prepare.py
lfv/pipeline/dino_bbox.py
lfv/pipeline/sam2_mask.py
lfv/pipeline/sample_points.py
lfv/pipeline/tracking.py
lfv/pipeline/se3_trajectory.py
lfv/pipeline/hand_bbox.py
lfv/pipeline/hand_mask.py
lfv/pipeline/contact_timing.py
lfv/pipeline/contact_heatmap.py
lfv/pipeline/dinov2_features.py
lfv/pipeline/hamer_hand_pose.py
lfv/pipeline/thumb_index_grasp_label.py
```

Batch helpers:

```text
scripts/run_hand_pouring_contact_batch.sh
scripts/run_hand_pouring_grasp_batch.sh
```

Validation helpers:

```text
tools/check_contact_field_outputs.py
tools/check_hand_pouring_contact_batch.py
tools/check_hand_pouring_grasp_batch.py
tools/check_pipeline_outputs.py
```

Key visualization / validation tools:

```text
tools/verify_episode0_graspnet_contact_roi.py
tools/visualize_hamer_thumb_index_grasp_open3d.py
tools/visualize_episode0_hamer_thumb_index_grasp_open3d.py
scripts/visualize_episode0_graspnet_contact_roi.sh
```

## Removed From Current Repository

Removed intentionally:

```text
data/
diffusion_policy_3d/
lfv_sim/
lfv_sim.zip
scripts/model/
scripts/sim/
scripts/sim_calibration/
scripts/legacy/
configs/model/
configs/paths/
utils/
old run_step*.py wrappers
old model/simulation/training docs
```

Reason:

- datasets and outputs should stay under `/media/ljian/lj`;
- model code will be rewritten for the new two-stage framework;
- simulation and legacy training code should not constrain the new design;
- all removed code is still available in the external lightweight backup.

## Third-Party Code Policy

Third-party dependencies are retained because data processing still depends on
them:

```text
third_party/dinov2_weights
third_party/hamer -> /home/users1/ljian/hamer
third_party/sam2
third_party/tapip3d
```

Do not edit third-party source directly unless explicitly required. Wrap it from
LFV modules instead.

HaMeR should run through:

```text
scripts/run_hamer_demo_env.sh
```

which uses:

```text
/home/users1/ljian/anaconda3/envs/hamer/bin/python
```

## New Model Architecture

The future model design is a two-stage multimodal generation framework.

### Stage 1: Contact-Grasp Generation

Package:

```text
lfv/models/contact_grasp_generation/
lfv/training/contact_grasp/
lfv/inference/contact_grasp/
lfv/evaluation/contact_grasp/
```

Inputs:

- manipulated object point cloud;
- point normals;
- point DINO semantic features;
- optional visibility/scale/task metadata.

Outputs:

- point-wise task contact heat field;
- robot parallel gripper grasp candidates;
- grasp width, contact points, confidence and quality.

Supervision already available:

```text
contact_heatmap/contact_heatmap.npz
dinov2_features/point_dinov2_features.npy
hamer_grasp_pseudo_label/grasp_pseudo_label.npz
```

### Stage 2: Functional Motion Generation

Package:

```text
lfv/models/functional_motion_generation/
lfv/training/functional_motion/
lfv/inference/functional_motion/
lfv/evaluation/functional_motion/
```

Inputs:

- manipulated object point cloud/features;
- reference or target object point cloud/features;
- object-object geometry relation;
- task metadata.

Outputs:

- object-centric multimodal SE(3) functional motion trajectories.

### Robot Feasibility Selection

Package:

```text
lfv/robot/
```

Responsibilities:

- combine grasp samples and motion samples;
- convert object motion to TCP trajectory;
- run continuous IK;
- score joint deltas, joint limits, smoothness and collision;
- select executable grasp-motion pair.

## Current Package Skeleton

```text
lfv/data_processing/              preprocessing API and episode I/O helpers
lfv/datasets/                     model datasets
lfv/models/common/                shared point/SE(3)/diffusion modules
lfv/models/contact_grasp_generation/
lfv/models/functional_motion_generation/
lfv/training/
lfv/inference/
lfv/evaluation/
lfv/robot/
lfv/tasks/
lfv/visualization/
```

The skeleton contains README files and minimal `__init__.py` files only.
Model implementation should start from these directories. The exception is
`lfv/data_processing/episode_io.py`, which already contains the retained
episode read/write helpers used by the current preprocessing pipeline.

## Config Layout

Retained pipeline configs:

```text
configs/pipeline/hand_pouring.yaml
configs/pipeline/picknplace.yaml
configs/pipeline/contact_field.yaml
```

New data and experiment configs:

```text
configs/data/datasets/hand_pouring_lfv.yaml
configs/data/tasks/hand_pouring.yaml
configs/experiments/contact_grasp/base.yaml
configs/experiments/functional_motion/base.yaml
configs/experiments/two_stage_pipeline/base.yaml
```

`configs/experiments/*/base.yaml` files are placeholders. They define the
intended structure but not final model hyperparameters.

## Script Layout

Stable current entrypoints:

```text
scripts/run_pipeline.py
scripts/run_hand_pouring_contact_batch.sh
scripts/run_hand_pouring_grasp_batch.sh
scripts/run_hamer_demo_env.sh
```

Future wrappers:

```text
scripts/preprocess/
scripts/train/
scripts/infer/
scripts/evaluate/
scripts/visualize/
scripts/robot/
```

Rule:

- implementation belongs in `lfv/`;
- `scripts/` should be thin command-line wrappers;
- `tools/` is for debugging and inspection.

## Immediate Next Steps

1. Implement grasp geometry diagnostics before training any model:

```text
lfv/evaluation/contact_grasp/grasp_geometry.py
scripts/evaluate/diagnose_thumb_index_grasp.py
```

This should answer whether the current thumb-index grasp really places the cup
handle inside the gripper volume.

2. Improve grasp `approach` selection:

Current issue from hand-pouring:

```text
approach = -surface_normal
```

can bias toward camera depth. Replace it with candidate scoring:

```text
top_down
surface_normal
camera_to_object
rotations_about_closing
```

3. Implement the first minimal dataset:

```text
lfv/datasets/contact_grasp_dataset.py
```

It should read processed artifacts only and not run segmentation/tracking.

4. Build a small contact heat baseline before a full diffusion model:

- input: points + normals + DINO;
- output: contact heat;
- optional grasp pseudo-label regression/scoring.

5. Start functional motion model after the contact/grasp data interface is
stable.

## Rules For Future Work

- Do not write datasets, generated labels, checkpoints or visual outputs into
  the repository.
- Use `/media/ljian/lj` for data and model outputs.
- Keep third-party source under `third_party/`.
- Keep preprocessing separate from model datasets.
- Do not resurrect old model code into the new structure wholesale.
- Prefer small validated baselines over rebuilding the full diffusion stack at
  once.
