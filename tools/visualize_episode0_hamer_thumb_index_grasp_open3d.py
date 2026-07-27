#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib

import matplotlib.cm as cm
import numpy as np
import open3d as o3d


DEFAULT_EPISODE = pathlib.Path("/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0")


def grasp_points_from_T(T: np.ndarray, *, width: float, depth: float = 0.045) -> np.ndarray:
    depth_base = 0.02
    local = np.asarray(
        [
            [-depth_base, -width / 2, 0.0],
            [depth, -width / 2, 0.0],
            [-depth_base, width / 2, 0.0],
            [depth, width / 2, 0.0],
            [0.0, 0.0, 0.0],
            [depth, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return (T[:3, :3] @ local.T).T + T[:3, 3][None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Open3D viewer for episode_0 HaMeR thumb-index grasp pseudo label.")
    parser.add_argument("--episode-dir", default=str(DEFAULT_EPISODE))
    parser.add_argument("--label-dir", default=None)
    args = parser.parse_args()

    ep = pathlib.Path(args.episode_dir)
    label_dir = pathlib.Path(args.label_dir) if args.label_dir else ep / "hamer_grasp_pseudo_label"
    heat_data = np.load(ep / "contact_heatmap" / "contact_heatmap.npz")
    label = np.load(label_dir / "grasp_pseudo_label.npz")

    points = heat_data["points_camera"].astype(np.float64)
    heat = heat_data["contact_heat"].astype(np.float64)
    T = label["T_grasp_cam"].astype(np.float64)
    width = float(label["width_m"])
    q_thumb = label["q_thumb_cam"].astype(np.float64)
    q_index = label["q_index_cam"].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(cm.get_cmap("magma")(np.clip(heat, 0, 1))[:, :3])

    gripper_pts = grasp_points_from_T(T, width=width)
    lines = np.asarray([[0, 1], [2, 3], [0, 2], [4, 5]], dtype=np.int32)
    gripper = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(gripper_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    gripper.colors = o3d.utility.Vector3dVector(
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.25, 1.0]], dtype=np.float64)
    )

    thumb = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    thumb.paint_uniform_color([1.0, 0.0, 1.0])
    thumb.translate(q_thumb)
    index = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    index.paint_uniform_color([0.0, 1.0, 0.0])
    index.translate(q_index)
    tcp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=T[:3, 3])

    print(f"label_dir: {label_dir}")
    print(f"selected_frame: {int(label['selected_frame'])}")
    print(f"width_m: {width:.4f}, confidence: {float(label['confidence']):.3f}")
    o3d.visualization.draw_geometries(
        [pcd, gripper, thumb, index, tcp_frame],
        window_name="episode_0 HaMeR thumb-index grasp pseudo label",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
