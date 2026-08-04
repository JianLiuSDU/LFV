#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.visualization import render_topdown_grasp_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output = render_topdown_grasp_summary(
        lift_overlay_path=output_dir / "transferred_heat_lift_overlay.png",
        complete_heat_path=output_dir / "contact_full_camera_closeup.png",
        selected_rgb_path=output_dir / "graspnet_selected_rgb_clean.png",
        selected_open3d_path=output_dir / "graspnet_selected_open3d.png",
        grasp_report_path=output_dir / "graspnet_full_contact_report.json",
        output_path=output_dir / "topdown_grasp_summary.png",
    )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
