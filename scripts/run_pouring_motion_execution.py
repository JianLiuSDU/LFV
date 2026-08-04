#!/usr/bin/env python3
"""Fixed quick-iteration entry point for learned pouring execution and video."""

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


def _run(name: str, command: list[str], output_root: Path, env: dict[str, str]) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"\n========== {name} ==========", flush=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
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
    parser.add_argument(
        "--config",
        default="configs/experiments/functional_motion/pouring_cup_far_execution.yaml",
    )
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_root = Path(cfg.paths.output_root).expanduser().resolve()
    snapshot_dir = output_root / "snapshot"
    transfer_dir = output_root / "transfer"
    motion_dir = output_root / "motion_inference"
    execution_dir = Path(
        cfg.paths.get("execution_output_dir", output_root / "execution")
    ).expanduser().resolve()
    for path in (snapshot_dir, transfer_dir, motion_dir, execution_dir):
        path.mkdir(parents=True, exist_ok=True)

    process_env = os.environ.copy()
    process_env["CUDA_VISIBLE_DEVICES"] = str(cfg.runtime.gpu)
    if not args.skip_snapshot:
        command = [
            str(cfg.runtime.maniskill_python),
            "scripts/sim/export_pouring_contact_snapshot.py",
            "--output-dir",
            str(snapshot_dir),
        ]
        for key, value in cfg.scene.items():
            _append(command, key, value)
        _run("00_snapshot_far_scene", command, output_root, process_env)

    if not args.skip_transfer:
        _run(
            "01_soft_heatmap_transfer",
            [
                str(cfg.runtime.tapip_python),
                "scripts/affordance_transfer/transfer_contact_heatmap.py",
                "--config",
                str(cfg.paths.transfer_config),
                "--output-dir",
                str(transfer_dir),
            ],
            output_root,
            process_env,
        )

    if not args.skip_motion:
        command = [
            str(cfg.runtime.motion_python),
            "scripts/inference/infer_pouring_motion.py",
            "--snapshot",
            str(snapshot_dir / "pouring_snapshot.npz"),
            "--transfer-result",
            str(transfer_dir / "transfer_result.npz"),
            "--output-dir",
            str(motion_dir),
            "--model-repo",
            str(cfg.paths.model_repo),
            "--goal-checkpoint",
            str(cfg.paths.goal_checkpoint),
            "--trajectory-checkpoint",
            str(cfg.paths.trajectory_checkpoint),
            "--language-embedding",
            str(cfg.paths.language_embedding),
        ]
        for key, value in cfg.motion.items():
            _append(command, key, value)
        _run("02_trained_two_stage_motion", command, output_root, process_env)

    if not args.skip_execution:
        command = [
            str(cfg.runtime.maniskill_python),
            "scripts/robot/execute_pouring_motion_maniskill.py",
            "--snapshot-report",
            str(snapshot_dir / "snapshot_report.json"),
            "--motion-prediction",
            str(motion_dir / "pouring_motion_prediction.npz"),
            "--grasp-object",
            str(cfg.paths.grasp_object),
            "--output-dir",
            str(execution_dir),
        ]
        for key, value in cfg.execution.items():
            _append(command, key, value)
        _run("03_maniskill_execution_video", command, output_root, process_env)

    print(f"\nComplete quick-iteration run: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
