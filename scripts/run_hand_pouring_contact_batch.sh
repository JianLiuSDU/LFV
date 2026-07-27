#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/pipeline/hand_pouring.yaml}"
TAPIP3D_PY="${TAPIP3D_PY:-/home/users1/ljian/anaconda3/envs/tapip3d/bin/python}"
SAM2_PY="${SAM2_PY:-/home/users1/ljian/anaconda3/envs/sam2/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OVERWRITE=0
RUN_CHECK=1
EPISODES=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_hand_pouring_contact_batch.sh [options]

Options:
  --episodes IDS...   Process selected episodes only, e.g. --episodes 0 1 2
  --overwrite         Recompute existing outputs.
  --no-check          Skip final summary check.
  -h, --help          Show this help.

Environment overrides:
  CONFIG              Pipeline config path.
  TAPIP3D_PY          Python used for DINO/timing/DINOv2/contact_heatmap.
  SAM2_PY             Python used for SAM2 hand masks.
  HF_ENDPOINT         Hugging Face endpoint for GroundingDINO metadata/cache.

Default behavior processes all episodes and skips existing outputs.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EPISODES+=("$1")
        shift
      done
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --no-check)
      RUN_CHECK=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[batch] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

COMMON_ARGS=(--config "${CONFIG}")
if [[ ${#EPISODES[@]} -gt 0 ]]; then
  COMMON_ARGS+=(--episodes "${EPISODES[@]}")
fi
if [[ "${OVERWRITE}" -eq 1 ]]; then
  COMMON_ARGS+=(--overwrite)
fi

echo "[batch] root: ${ROOT_DIR}"
echo "[batch] config: ${CONFIG}"
echo "[batch] tapip3d python: ${TAPIP3D_PY}"
echo "[batch] sam2 python: ${SAM2_PY}"
if [[ ${#EPISODES[@]} -gt 0 ]]; then
  echo "[batch] episodes: ${EPISODES[*]}"
else
  echo "[batch] episodes: all"
fi
echo "[batch] overwrite: ${OVERWRITE}"

echo "[batch] stage 1/5: hand_bbox"
HF_ENDPOINT="${HF_ENDPOINT}" "${TAPIP3D_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps hand_bbox

echo "[batch] stage 2/5: hand_mask"
"${SAM2_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps hand_mask

echo "[batch] stage 3/5: timing"
"${TAPIP3D_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps timing

echo "[batch] stage 4/5: dinov2"
"${TAPIP3D_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps dinov2

echo "[batch] stage 5/5: contact_heatmap"
"${TAPIP3D_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps contact_heatmap

if [[ "${RUN_CHECK}" -eq 1 ]]; then
  echo "[batch] summary check"
  CHECK_ARGS=(--config "${CONFIG}")
  for ep in "${EPISODES[@]}"; do
    CHECK_ARGS+=(--episode "${ep}")
  done
  "${TAPIP3D_PY}" tools/check_hand_pouring_contact_batch.py "${CHECK_ARGS[@]}"
fi

echo "[batch] done"
