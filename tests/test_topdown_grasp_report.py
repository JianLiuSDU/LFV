import json
from pathlib import Path

from PIL import Image

from lfv.visualization import (
    render_instance_generalization_comparison,
    render_topdown_grasp_summary,
)


def test_topdown_grasp_report_saves_fixed_four_panel_image(tmp_path: Path):
    image_paths = []
    for index in range(4):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (80, 60), (20 * index, 60, 100)).save(path)
        image_paths.append(path)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "selected": {
                    "approach_to_desired_angle_deg": 5.0,
                    "contact_pair_width_m": 0.014,
                    "left_tip_heat": 0.8,
                    "right_tip_heat": 0.9,
                    "collision_part_ious": {
                        "global_iou": 0.0,
                        "left_finger_iou": 0.0,
                        "right_finger_iou": 0.0,
                        "palm_iou": 0.0,
                        "approach_path_iou": 0.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output = render_topdown_grasp_summary(
        lift_overlay_path=image_paths[0],
        complete_heat_path=image_paths[1],
        selected_rgb_path=image_paths[2],
        selected_open3d_path=image_paths[3],
        grasp_report_path=report_path,
        output_path=tmp_path / "summary.png",
    )
    assert output.is_file()
    with Image.open(output) as image:
        assert image.width > 1000
        assert image.height > 700


def test_instance_comparison_saves_fixed_two_by_two_image(tmp_path: Path):
    images = []
    for index in range(4):
        path = tmp_path / f"comparison_{index}.png"
        Image.new("RGB", (100, 70), (30 * index, 80, 120)).save(path)
        images.append(path)
    transfer_reports = []
    grasp_reports = []
    for index in range(2):
        transfer_path = tmp_path / f"transfer_{index}.json"
        transfer_path.write_text(
            json.dumps({"confidence": {"global": 0.3, "cycle": 0.1}}),
            encoding="utf-8",
        )
        transfer_reports.append(transfer_path)
        grasp_path = tmp_path / f"grasp_{index}.json"
        grasp_path.write_text(
            json.dumps(
                {
                    "selected": {
                        "approach_to_desired_angle_deg": 5.0,
                        "left_tip_heat": 0.8,
                        "right_tip_heat": 0.9,
                        "collision_part_ious": {"global_iou": 0.0},
                    }
                }
            ),
            encoding="utf-8",
        )
        grasp_reports.append(grasp_path)
    output = render_instance_generalization_comparison(
        baseline_label="baseline",
        candidate_label="candidate",
        baseline_heat_path=images[0],
        candidate_heat_path=images[1],
        baseline_grasp_path=images[2],
        candidate_grasp_path=images[3],
        baseline_transfer_report_path=transfer_reports[0],
        candidate_transfer_report_path=transfer_reports[1],
        baseline_grasp_report_path=grasp_reports[0],
        candidate_grasp_report_path=grasp_reports[1],
        output_path=tmp_path / "comparison.png",
    )
    assert output.is_file()
    with Image.open(output) as image:
        assert image.width > 1000
        assert image.height > 700
