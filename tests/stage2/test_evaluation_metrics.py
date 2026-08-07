import torch

from lfv.evaluation.functional_motion import trajectory_best_of_k_metrics
from lfv.geometry import identity_pose9d


def test_trajectory_best_of_k_selects_matching_candidate():
    target = identity_pose9d(1, 64)
    predicted = identity_pose9d(1, 2, 1, 64)
    predicted[:, 0, 0, :, 0] = 0.2
    metrics = trajectory_best_of_k_metrics(predicted, target)
    assert metrics["trajectory_top1_translation_m"] > 0.19
    assert metrics["trajectory_best_translation_m"] < 1e-6
