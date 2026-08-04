#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.utils.config import load_config


def _run_step(
    name: str,
    command: list[str],
    *,
    log_dir: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"\n========== {name} ==========", flush=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
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
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Step {name!r} failed; inspect {log_path}")


def _append_value(command: list[str], name: str, value: Any) -> None:
    option = f"--{name.replace('_', '-')}"
    if isinstance(value, (list, tuple)):
        command.append(option)
        command.extend(str(item) for item in value)
    else:
        command.extend([option, str(value)])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Soft Heatmap AffCorrs -> RGB-D lifting -> complete-surface "
            "antipodal completion -> top-down collision-free GraspNet."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/affordance_grasp/episode0_to_maniskill_topdown.yaml",
    )
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-lifting", action="store_true")
    parser.add_argument("--skip-graspnet", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tapip_python = str(cfg.runtime.tapip_python)
    graspnet_python = str(cfg.runtime.graspnet_python)
    transfer_output = Path(cfg.paths.transfer_output_dir).expanduser().resolve()
    output_dir = Path(cfg.paths.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "pipeline_logs"
    snapshot = Path(cfg.paths.snapshot).expanduser().resolve()
    transfer_result = transfer_output / "transfer_result.npz"
    contact_3d = output_dir / "transferred_contact_3d.npz"
    grasp_env = os.environ.copy()
    grasp_env["CUDA_VISIBLE_DEVICES"] = str(cfg.runtime.gpu)

    snapshot_export = cfg.get("snapshot_export")
    if snapshot_export and not args.skip_snapshot:
        command = [
            str(cfg.runtime.maniskill_python),
            "scripts/sim/export_pouring_contact_snapshot.py",
            "--output-dir",
            str(snapshot.parent),
        ]
        for key, value in snapshot_export.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key.replace('_', '-')}")
            elif isinstance(value, (list, tuple)):
                command.append(f"--{key.replace('_', '-')}")
                command.extend(str(item) for item in value)
            else:
                _append_value(command, key, value)
        _run_step("00_export_snapshot", command, log_dir=log_dir, env=grasp_env)

    if not args.skip_transfer:
        _run_step(
            "01_transfer_2d",
            [
                tapip_python,
                "scripts/affordance_transfer/transfer_contact_heatmap.py",
                "--config",
                str(cfg.paths.transfer_config),
                "--output-dir",
                str(transfer_output),
            ],
            log_dir=log_dir,
            env=os.environ.copy(),
        )
    if not args.skip_lifting:
        command = [
            tapip_python,
            "scripts/sim/lift_transferred_heat_to_complete_surface.py",
            "--transfer-result",
            str(transfer_result),
            "--snapshot",
            str(snapshot),
            "--output-dir",
            str(output_dir),
        ]
        for key, value in cfg.lifting.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key.replace('_', '-')}")
            else:
                _append_value(command, key, value)
        _run_step("02_lift_complete_surface", command, log_dir=log_dir)
    if not args.skip_graspnet:
        command = [
            "xvfb-run",
            "-a",
            graspnet_python,
            "scripts/sim/generate_graspnet_from_full_contact.py",
            "--input",
            str(contact_3d),
            "--snapshot",
            str(snapshot),
            "--graspnet-root",
            str(cfg.paths.graspnet_root),
            "--checkpoint",
            str(cfg.paths.graspnet_checkpoint),
        ]
        for key, value in cfg.graspnet.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key.replace('_', '-')}")
            else:
                _append_value(command, key, value)
        _run_step("03_graspnet_topdown", command, log_dir=log_dir, env=grasp_env)

    visualization = cfg.visualization
    _run_step(
        "04_render_complete_heat",
        [
            "xvfb-run",
            "-a",
            graspnet_python,
            "scripts/sim/render_pouring_contact_camera_view.py",
            "--input",
            str(contact_3d),
            "--snapshot",
            str(snapshot),
            "--output-dir",
            str(output_dir),
            "--heat",
            "full",
            "--point-size",
            str(visualization.point_size),
            "--render-scale",
            str(visualization.render_scale),
            "--closeup-size",
            str(visualization.closeup_size),
        ],
        log_dir=log_dir,
        env=grasp_env,
    )
    _run_step(
        "05_compose_summary",
        [
            tapip_python,
            "scripts/sim/compose_topdown_grasp_summary.py",
            "--output-dir",
            str(output_dir),
        ],
        log_dir=log_dir,
    )

    grasp_report = json.loads(
        (output_dir / "graspnet_full_contact_report.json").read_text(
            encoding="utf-8"
        )
    )
    lifting_report = json.loads(
        (output_dir / "transferred_contact_3d_report.json").read_text(
            encoding="utf-8"
        )
    )
    transfer_report_path = transfer_output / "transfer_report.json"
    snapshot_report_path = snapshot.parent / "snapshot_report.json"
    pipeline_report = {
        "experiment_name": cfg.experiment_name,
        "config": str(Path(args.config).expanduser().resolve()),
        "snapshot": json.loads(snapshot_report_path.read_text(encoding="utf-8")),
        "transfer": json.loads(transfer_report_path.read_text(encoding="utf-8")),
        "lifting": lifting_report,
        "grasp": grasp_report,
        "outputs": {
            "transfer_summary": str(transfer_output / "transfer_summary.png"),
            "summary": str(output_dir / "topdown_grasp_summary.png"),
            "selected_grasp_camera": str(output_dir / "graspnet_selected.npy"),
            "selected_grasp_world": str(
                output_dir / "graspnet_selected_world.npy"
            ),
            "selected_grasp_object": str(
                output_dir / "graspnet_selected_object.npy"
            ),
        },
    }
    report_path = output_dir / "topdown_grasp_pipeline_report.json"
    report_path.write_text(
        json.dumps(pipeline_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nPipeline complete: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
