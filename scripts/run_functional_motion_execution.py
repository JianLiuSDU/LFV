#!/usr/bin/env python3
"""Config-driven contact-transfer, grasp, motion-inference and execution runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.utils.config import load_config


def _append(command: list[str], key: str, value: Any) -> None:
    option = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            command.append(option)
    elif isinstance(value, (list, tuple)):
        command.append(option)
        command.extend(str(item) for item in value)
    else:
        command.extend((option, str(value)))


def _run(name: str, command: list[str], output: Path, env: dict[str, str]) -> None:
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"\n========== {name} ==========\n{' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-grasp", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg.paths.output_root).expanduser().resolve()
    snapshot_dir = output / "snapshot"
    transfer_dir = output / "affordance_transfer"
    motion_dir = output / "motion_inference"
    execution_dir = output / "execution"
    for path in (snapshot_dir, transfer_dir, motion_dir, execution_dir):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.runtime.gpu)

    if not args.skip_snapshot:
        command = [str(cfg.runtime.maniskill_python), "scripts/sim/export_task_snapshot.py",
                   "--task", str(cfg.task), "--output-dir", str(snapshot_dir)]
        for key, value in cfg.scene.items():
            _append(command, key, value)
        _run("00_snapshot", command, output, env)
    if not args.skip_transfer:
        _run("01_affordance_transfer", [str(cfg.runtime.tapip_python),
             "scripts/affordance_transfer/transfer_contact_heatmap.py", "--config",
             str(cfg.paths.transfer_config), "--output-dir", str(transfer_dir)], output, env)
    if not args.skip_grasp:
        _run("02_complete_surface_grasp", [str(cfg.runtime.tapip_python),
             "scripts/sim/run_transferred_heat_topdown_grasp.py", "--config",
             str(cfg.paths.grasp_config), "--skip-snapshot", "--skip-transfer"], output, env)
    if not args.skip_motion:
        command = [str(cfg.runtime.motion_python), "scripts/inference/infer_functional_motion.py",
                   "--task", str(cfg.task), "--snapshot", str(snapshot_dir / "task_snapshot.npz"),
                   "--transfer-result", str(transfer_dir / "transfer_result.npz"),
                   "--output-dir", str(motion_dir), "--model-repo", str(cfg.paths.model_repo),
                   "--goal-checkpoint", str(cfg.paths.goal_checkpoint),
                   "--trajectory-checkpoint", str(cfg.paths.trajectory_checkpoint),
                   "--language-embedding", str(cfg.paths.language_embedding)]
        for key, value in cfg.motion.items():
            _append(command, key, value)
        _run("03_goalpose_full64_inference", command, output, env)
    if not args.skip_execution:
        command = [str(cfg.runtime.maniskill_python),
                   "scripts/robot/execute_functional_motion_maniskill.py",
                   "--task", str(cfg.task), "--snapshot-report", str(snapshot_dir / "snapshot_report.json"),
                   "--motion-prediction", str(motion_dir / "functional_motion_prediction.npz"),
                   "--grasp-object", str(cfg.paths.grasp_object), "--output-dir", str(execution_dir)]
        for key, value in cfg.execution.items():
            _append(command, key, value)
        _run("04_maniskill_execution", command, output, env)
    print(f"\nFunctional task complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
