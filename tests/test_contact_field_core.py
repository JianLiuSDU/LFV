from __future__ import annotations

import unittest

import numpy as np

from lfv.pipeline.contact_field import (
    _connected_components_high_heat,
    build_anchor_point_cloud,
    fit_elliptical_heatmap,
)


class ContactFieldCoreTest(unittest.TestCase):
    def test_anchor_point_cloud_keeps_pixel_correspondence(self):
        h, w = 48, 64
        depth = np.ones((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        mask[10:30, 15:45] = True
        intrinsics = np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1]], dtype=np.float32)
        pc = build_anchor_point_cloud(
            depth,
            mask,
            intrinsics,
            num_points=128,
            rng=np.random.default_rng(0),
            outlier_std_ratio=0,
            normal_k=8,
        )
        self.assertEqual(pc.points_camera.shape, (128, 3))
        self.assertEqual(pc.pixels_uv.shape, (128, 2))
        self.assertEqual(pc.normals_camera.shape, (128, 3))
        self.assertTrue(np.all(mask[pc.pixels_uv[:, 1], pc.pixels_uv[:, 0]]))
        self.assertTrue(np.all(np.isfinite(pc.points_object_m)))
        self.assertGreater(pc.object_scale, 0)

    def test_elliptical_heatmap_is_masked_and_assignable(self):
        h, w = 60, 80
        object_mask = np.zeros((h, w), dtype=bool)
        object_mask[10:50, 15:65] = True
        yy, xx = np.nonzero(object_mask)
        pixels = np.stack([xx, yy], axis=-1).astype(np.int32)
        center = np.array([38, 28])
        dist = np.linalg.norm(pixels - center[None, :], axis=1)
        evidence = np.exp(-(dist ** 2) / (2 * 5.0 ** 2)).astype(np.float32)
        heatmap, seed_mask, meta = fit_elliptical_heatmap(
            pixels,
            evidence,
            object_mask,
            seed_threshold=0.5,
            fallback_topk=16,
            min_axis_px=3,
            max_axis_px=20,
            cov_regularizer=1,
        )
        self.assertEqual(meta["status"], "ok")
        self.assertGreater(int(np.sum(seed_mask)), 3)
        self.assertLessEqual(float(np.max(heatmap)), 1.0)
        self.assertEqual(float(np.max(heatmap[~object_mask])), 0.0)
        point_heat = heatmap[pixels[:, 1], pixels[:, 0]]
        self.assertEqual(point_heat.shape[0], pixels.shape[0])

    def test_surface_correction_keeps_main_component(self):
        points = np.array(
            [[0.00, 0.00, 0], [0.01, 0.00, 0], [0.00, 0.01, 0], [0.50, 0.50, 0]],
            dtype=np.float32,
        )
        normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (4, 1))
        heat = np.array([1.0, 0.8, 0.7, 0.9], dtype=np.float32)
        corrected, meta = _connected_components_high_heat(
            points,
            normals,
            heat,
            threshold=0.4,
            k=3,
            max_neighbor_dist_m=0.03,
            min_normal_dot=0.0,
        )
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(int(np.argmax(corrected)), 0)
        self.assertEqual(float(corrected[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
