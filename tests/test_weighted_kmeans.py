import numpy as np

from lfv.affordance_transfer.clustering import weighted_kmeans


def test_weighted_kmeans_is_deterministic_and_separates_clusters():
    rng = np.random.default_rng(3)
    left = rng.normal(loc=(-1.0, 0.0), scale=0.03, size=(20, 2))
    right = rng.normal(loc=(1.0, 0.0), scale=0.03, size=(20, 2))
    features = np.concatenate([left, right], axis=0).astype(np.float32)
    weights = np.concatenate([np.ones(20), np.full(20, 3.0)])

    first = weighted_kmeans(features, 2, weights=weights, seed=7, n_init=3)
    second = weighted_kmeans(features, 2, weights=weights, seed=7, n_init=3)
    np.testing.assert_allclose(first.centers, second.centers)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.labels[:20].ptp() == 0
    assert first.labels[20:].ptp() == 0
    assert first.labels[0] != first.labels[-1]
    assert np.isclose(first.cluster_mass.sum(), weights.sum())
