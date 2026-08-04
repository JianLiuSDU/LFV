import numpy as np

from lfv.affordance_transfer.soft_affcorrs import soft_heatmap_affcorrs


def test_bidirectional_product_prefers_semantically_matching_target_region():
    source_features = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.0, 1.0],
            [0.05, 0.98],
        ],
        dtype=np.float32,
    )
    source_heat = np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32)
    target_features = np.array(
        [
            [1.0, 0.0],
            [0.97, 0.08],
            [0.0, 1.0],
            [0.08, 0.97],
        ],
        dtype=np.float32,
    )
    result = soft_heatmap_affcorrs(
        source_features,
        source_heat,
        target_features,
        source_clusters=1,
        target_clusters=2,
        positive_threshold=0.2,
        forward_temperature=0.08,
        backward_temperature=0.05,
        n_init=2,
    )
    patch_scores = result.target_cluster_scores[result.target_clustering.labels]
    assert patch_scores[:2].mean() > 20 * patch_scores[2:].mean()
    assert np.isclose(result.source_cluster_weights.sum(), 1.0)
    assert np.allclose(result.forward_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.backward_probabilities.sum(axis=1), 1.0)


def test_source_clustering_uses_only_heat_positive_patches():
    features = np.eye(4, dtype=np.float32)
    heat = np.array([1.0, 0.4, 0.1, 0.0], dtype=np.float32)
    result = soft_heatmap_affcorrs(
        features,
        heat,
        features,
        source_clusters=8,
        target_clusters=4,
        positive_threshold=0.2,
        n_init=1,
    )
    assert result.source_clustering.centers.shape[0] == 2
    np.testing.assert_array_equal(result.source_positive_mask, [True, True, False, False])
    assert 0 < result.retained_heat_mass < 1
