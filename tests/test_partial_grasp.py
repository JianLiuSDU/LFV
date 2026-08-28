import numpy as np

from lfv.deployment.partial_grasp import build_contact_pair_hypotheses, evaluate_contact_pair_against_full_cloud


def test_contact_pair_has_topdown_frame_and_virtual_side():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(128, 3)).astype(np.float32) * np.array([0.03, 0.02, 0.04], dtype=np.float32)
    points[:, 2] += 0.55
    heat = np.exp(-np.sum((points - np.array([0.0, 0.0, 0.55], dtype=np.float32)) ** 2, axis=1) / 0.002).astype(np.float32)
    hs = build_contact_pair_hypotheses(points, heat, top_k=4)
    assert len(hs) == 4
    for h in hs:
        assert h.tcp_camera.shape == (4, 4)
        assert np.allclose(h.tcp_camera[3], [0, 0, 0, 1])
        assert np.isclose(np.linalg.det(h.tcp_camera[:3, :3]), 1.0, atol=1e-4)
        assert h.metadata["virtual_second_contact"]


def test_contact_pair_oracle_reports_both_contacts():
    points = np.stack([np.array([-0.03, 0.0, 0.5]), np.array([0.03, 0.0, 0.5])], axis=0).astype(np.float32)
    hs = build_contact_pair_hypotheses(points, np.ones(2, dtype=np.float32), width_candidates_m=(0.06,), top_k=1)
    rows = evaluate_contact_pair_against_full_cloud(hs, points, support_radius_m=0.04)
    assert rows and rows[0]["first_supported"]

