from __future__ import annotations

import json

import numpy as np
from PIL import Image

from lfv.deployment.input_schema import load_camera_input
from lfv.deployment.output_schema import CameraPlanResult
from lfv.geometry.registration import rigid_icp_to_visible
from lfv.geometry.sam3d_completion import backproject_mask


def test_camera_input_contract(tmp_path):
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.ones((8, 10), dtype=np.float32)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    np.save(tmp_path / "depth.npy", depth)
    (tmp_path / "intrinsics.json").write_text(json.dumps({"fx": 10, "fy": 10, "cx": 5, "cy": 4}))
    Image.fromarray(np.ones((8, 10), dtype=np.uint8)).save(tmp_path / "cup_mask.png")
    Image.fromarray(np.ones((8, 10), dtype=np.uint8)).save(tmp_path / "bowl_mask.png")
    item = load_camera_input(tmp_path)
    assert item.rgb.shape == (8, 10, 3)
    assert item.validate()["valid_depth_ratio"] == 1.0


def test_backprojection_and_registration():
    mask = np.ones((2, 2), dtype=bool)
    depth = np.ones((2, 2), dtype=np.float32)
    points, pixels = backproject_mask(mask, depth, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
    assert points.shape == (4, 3) and pixels.shape == (4, 2)
    transform, aligned, rms = rigid_icp_to_visible(points, points + np.array([0.1, 0.0, 0.0], dtype=np.float32))
    assert aligned.shape == points.shape and np.isfinite(rms)
    assert np.allclose(transform[:3, 3], [0.1, 0.0, 0.0], atol=2e-2)


def test_camera_plan_validation():
    result = CameraPlanResult(np.eye(4), np.repeat(np.eye(4)[None], 3, axis=0), np.repeat(np.eye(4)[None], 3, axis=0))
    result.validate()
