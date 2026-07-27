#!/usr/bin/env bash
set -euo pipefail

cd /home/users1/ljian/LFV

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" tools/verify_episode0_graspnet_contact_roi.py --show-open3d "$@"
