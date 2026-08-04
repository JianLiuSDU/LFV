import numpy as np

from lfv.affordance_transfer.confidence import compute_transfer_confidence
from lfv.affordance_transfer.soft_affcorrs import soft_heatmap_affcorrs


def test_confidence_reports_all_three_diagnostics_and_rejection():
    features = np.array(
        [[1.0, 0.0], [0.99, 0.03], [0.0, 1.0], [0.02, 0.99]],
        dtype=np.float32,
    )
    matching = soft_heatmap_affcorrs(
        features,
        np.array([1.0, 0.7, 0.0, 0.0], dtype=np.float32),
        features,
        source_clusters=1,
        target_clusters=2,
        positive_threshold=0.2,
        n_init=1,
    )
    scores = matching.target_cluster_scores[matching.target_clustering.labels]
    confidence = compute_transfer_confidence(
        matching,
        scores,
        minimum_global_score=1.1,
    )
    assert {"global", "cycle", "peak", "entropy"} <= confidence.values.keys()
    assert not confidence.accepted
    assert "low_global_confidence" in confidence.rejection_reasons
