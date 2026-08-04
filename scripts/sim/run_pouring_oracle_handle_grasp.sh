#!/usr/bin/env bash
set -euo pipefail

LFV_ROOT=/home/users1/ljian/LFV
SNAPSHOT=/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/seed_0_dataset_aligned/pouring_snapshot.npz
OUTPUT_DIR=/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/seed_0_dataset_aligned/oracle_handle_upper_v2
ORACLE_INPUT="${OUTPUT_DIR}/oracle_handle_contact.npz"
CONTACT_PY=/home/users1/ljian/anaconda3/envs/tapip3d/bin/python
GRASPNET_PY=/home/users1/ljian/anaconda3/envs/graspnet/bin/python
GRASPNET_ROOT=/home/users1/ljian/graspnet-baseline
TASK_GPU="${LFV_GPU:-1}"

cd "${LFV_ROOT}"

"${CONTACT_PY}" scripts/sim/create_oracle_handle_contact.py \
  --snapshot "${SNAPSHOT}" \
  --output-dir "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${TASK_GPU}" xvfb-run -a "${GRASPNET_PY}" \
  scripts/sim/generate_graspnet_from_full_contact.py \
  --input "${ORACLE_INPUT}" \
  --snapshot "${SNAPSHOT}" \
  --graspnet-root "${GRASPNET_ROOT}" \
  --checkpoint checkpoint-rs.tar \
  --seed 0 \
  --num-points 25600 \
  --target-ratio 0.25 \
  --heat-threshold 0.10 \
  --collision-voxel 0.008 \
  --collision-threshold 0.14 \
  --max-candidates 300 \
  --max-decoded-before-refine 100 \
  --max-refined-before-collision 400 \
  --pairs-per-candidate 24 \
  --max-pair-distance 0.10 \
  --width-margin 0.006 \
  --max-gripper-width 0.080 \
  --visualize-top-k 1 \
  --no-hard-workspace-mask \
  --topdown \
  --topdown-max-angle-deg 15 \
  --pair-max-vertical-delta 0.003 \
  --min-both-tip-heat 0.30 \
  --max-tip-surface-distance 0.005

CUDA_VISIBLE_DEVICES="${TASK_GPU}" xvfb-run -a "${GRASPNET_PY}" \
  scripts/sim/render_pouring_contact_camera_view.py \
  --input "${ORACLE_INPUT}" \
  --snapshot "${SNAPSHOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --heat full \
  --point-size 9 \
  --render-scale 4 \
  --closeup-size 800

echo "Oracle handle validation outputs: ${OUTPUT_DIR}"
