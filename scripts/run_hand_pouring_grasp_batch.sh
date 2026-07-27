#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/pipeline/hand_pouring.yaml}"
GRASP_PY="${GRASP_PY:-${TAPIP3D_PY:-/home/users1/ljian/anaconda3/envs/tapip3d/bin/python}}"

OVERWRITE=0
RUN_CHECK=1
EPISODES=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_hand_pouring_grasp_batch.sh [options]

Options:
  --episodes IDS...   Process selected episodes only, e.g. --episodes 0 1 2
  --overwrite         Recompute existing outputs (re-runs HaMeR and labels).
  --no-check          Skip final summary check.
  -h, --help          Show this help.

Environment overrides:
  CONFIG              Pipeline config path.
  GRASP_PY            Python used for the hamer/thumb_index_grasp pipeline stages.
                      Defaults to TAPIP3D_PY, then the tapip3d conda env.
  TAPIP3D_PY          Fallback python for GRASP_PY.
  HAMER_PYTHON        Python used inside scripts/run_hamer_demo_env.sh (HaMeR env).

Stages:
  1. hamer              export contact-window RGB frames and run HaMeR once for
                        all episodes missing 2D hand keypoints.
  2. thumb_index_grasp  build parallel-gripper pseudo labels from keypoints.

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
echo "[batch] grasp python: ${GRASP_PY}"
if [[ ${#EPISODES[@]} -gt 0 ]]; then
  echo "[batch] episodes: ${EPISODES[*]}"
else
  echo "[batch] episodes: all"
fi
echo "[batch] overwrite: ${OVERWRITE}"

echo "[batch] stage 1/2: hamer"
"${GRASP_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps hamer

echo "[batch] stage 2/2: thumb_index_grasp"
"${GRASP_PY}" scripts/run_pipeline.py "${COMMON_ARGS[@]}" --steps thumb_index_grasp

if [[ "${RUN_CHECK}" -eq 1 ]]; then
  echo "[batch] summary check"
  CHECK_ARGS=(--config "${CONFIG}")
  for ep in "${EPISODES[@]}"; do
    CHECK_ARGS+=(--episode "${ep}")
  done
  "${GRASP_PY}" tools/check_hand_pouring_grasp_batch.py "${CHECK_ARGS[@]}"
fi

echo "[batch] done"
