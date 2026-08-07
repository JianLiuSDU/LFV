import numpy as np

from lfv.datasets.functional_motion.sampling import (
    assert_unique_aligned,
    farthest_pixel_sample,
)


def test_unique_farthest_sampling_and_alignment():
    v, u = np.mgrid[:30, :40]
    candidates = np.stack((u.reshape(-1), v.reshape(-1)), axis=-1)
    pixels = farthest_pixel_sample(candidates, 256)
    points = np.concatenate((pixels.astype(np.float32), np.ones((256, 1))), axis=1)
    dino = np.repeat(pixels.astype(np.float32), 4, axis=1)
    assert_unique_aligned(pixels, points, dino, 256)
    assert np.unique(pixels, axis=0).shape[0] == 256
