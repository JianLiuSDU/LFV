#!/usr/bin/env bash
set -euo pipefail

HAMER_ROOT="${HAMER_ROOT:-/home/users1/ljian/LFV/third_party/hamer}"
HAMER_PYTHON="${HAMER_PYTHON:-/home/users1/ljian/anaconda3/envs/hamer/bin/python}"

cd "${HAMER_ROOT}"
exec "${HAMER_PYTHON}" demo.py "$@"
