from __future__ import annotations

import json

import numpy as np

from lfv.datasets.functional_motion.source_io import load_episode_calibration


def test_current_meta_calibration(tmp_path):
    meta = {
        "depth_scale": 0.001,
        "depth_intrinsics_original": {
            "fx": 500.0,
            "fy": 501.0,
            "ppx": 320.0,
            "ppy": 240.0,
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    scale, intrinsic, source = load_episode_calibration(tmp_path)
    assert scale == 0.001
    np.testing.assert_allclose(
        intrinsic,
        [[500.0, 0.0, 320.0], [0.0, 501.0, 240.0], [0.0, 0.0, 1.0]],
    )
    assert source == "meta.json"


def test_legacy_tracking_calibration_is_metric(tmp_path):
    tracking = tmp_path / "point_tracking"
    tracking.mkdir()
    intrinsic = np.asarray(
        [[617.159, 0.0, 320.0], [0.0, 617.159, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savez_compressed(tracking / "tapip3d_result.npz", intrinsics=intrinsic[None])
    scale, loaded, source = load_episode_calibration(tmp_path)
    assert scale == 1.0
    np.testing.assert_allclose(loaded, intrinsic)
    assert source == "tapip3d_result.npz"
