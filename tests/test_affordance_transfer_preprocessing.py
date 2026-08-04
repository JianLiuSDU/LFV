import numpy as np

from lfv.affordance_transfer.preprocessing import (
    map_grid_to_original,
    prepare_image,
)


def test_letterbox_mapping_round_trip_and_mask_alignment():
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[20:60, 40:90] = (180, 80, 20)
    mask = np.zeros((80, 120), dtype=bool)
    mask[20:60, 40:90] = True
    heat = np.zeros_like(mask, dtype=np.float32)
    heat[30:40, 45:55] = 1.0

    prepared = prepare_image(
        rgb, mask, heatmap=heat, input_size=56, patch_size=14, bbox_margin=0.1
    )
    assert prepared.rgb.shape == (56, 56, 3)
    assert prepared.mask.shape == (56, 56)
    assert prepared.heatmap.shape == (56, 56)
    points = np.array([[40.0, 20.0], [89.0, 59.0], [52.5, 33.25]])
    reconstructed = prepared.transform.input_to_original(
        prepared.transform.original_to_input(points)
    )
    np.testing.assert_allclose(reconstructed, points, atol=1e-5)

    grid = np.zeros((4, 4), dtype=np.float32)
    grid[1, 1] = 1.0
    mapped = map_grid_to_original(grid, prepared.transform, original_mask=mask)
    assert mapped.shape == mask.shape
    assert np.all(mapped[~mask] == 0)
    assert float(mapped.max()) > 0


def test_prepare_rejects_non_divisible_input_size():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    try:
        prepare_image(rgb, mask, input_size=50, patch_size=14)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected a divisibility error.")
