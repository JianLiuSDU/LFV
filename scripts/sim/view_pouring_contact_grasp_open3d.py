#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


HEAT_KEYS = {
    "full": "full_heat",
    "visible": "projected_visible_heat",
    "pair_visible": "pair_visible_heat",
    "opposite": "opposite_heat",
}


def _heat_colors(heat: np.ndarray) -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32).reshape(-1)
    scale = max(float(np.quantile(heat, 0.995)), 1e-6)
    normalized = np.clip(heat / scale, 0.0, 1.0)
    heat_u8 = np.round(normalized * 255.0).astype(np.uint8)
    turbo_bgr = cv2.applyColorMap(heat_u8[:, None], cv2.COLORMAP_TURBO)[:, 0]
    turbo_rgb = turbo_bgr[:, ::-1].astype(np.float64) / 255.0
    alpha = np.clip(np.sqrt(normalized)[:, None], 0.08, 1.0)
    return (1.0 - alpha) * 0.58 + alpha * turbo_rgb


def _grasp_points(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    width = float(max(row[1], 0.015))
    depth = float(max(row[3], 0.025))
    rotation = row[4:13].reshape(3, 3)
    center = row[13:16]
    local = np.asarray(
        [
            [-0.020, -width / 2, 0],
            [depth, -width / 2, 0],
            [-0.020, width / 2, 0],
            [depth, width / 2, 0],
            [0, 0, 0],
            [depth, 0, 0],
        ],
        dtype=np.float32,
    )
    return (rotation @ local.T).T + center[None]


def _cylinder_between(
    start: np.ndarray,
    end: np.ndarray,
    *,
    radius: float,
    color: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    delta = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(delta))
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=max(length, 1e-6),
        resolution=18,
    )
    direction = delta / max(length, 1e-8)
    cross = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    cross_norm = float(np.linalg.norm(cross))
    cosine = float(np.clip(direction[2], -1.0, 1.0))
    if cross_norm < 1e-8:
        axis_angle = (
            np.zeros(3, dtype=np.float64)
            if cosine >= 0
            else np.array([np.pi, 0.0, 0.0], dtype=np.float64)
        )
    else:
        axis_angle = cross / cross_norm * np.arccos(cosine)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    transform[:3, 3] = 0.5 * (np.asarray(start) + np.asarray(end))
    cylinder.transform(transform)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color)
    return cylinder


def _grasp_geometries(
    rows: np.ndarray,
    top_k: int,
) -> list[o3d.geometry.TriangleMesh]:
    geometries: list[o3d.geometry.TriangleMesh] = []
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 17)
    for index, row in enumerate(rows[:top_k]):
        points = _grasp_points(row)
        color = (
            np.array([1.0, 0.05, 0.05])
            if index == 0
            else np.array([0.10, 0.85, 0.20])
        )
        radius = 0.0024 if index == 0 else 0.0013
        for start, end in ((0, 1), (2, 3), (0, 2), (4, 5)):
            geometries.append(
                _cylinder_between(
                    points[start],
                    points[end],
                    radius=radius,
                    color=color,
                )
            )
    return geometries


def _pair_lines(
    points: np.ndarray,
    pairs: np.ndarray,
    count: int,
) -> o3d.geometry.LineSet | None:
    if count <= 0 or not len(pairs):
        return None
    pairs = np.asarray(pairs, dtype=np.float32).reshape(-1, 9)
    order = np.argsort(pairs[:, 2])[::-1][:count]
    endpoints = pairs[order, :2].astype(np.int64)
    line_points = points[endpoints].reshape(-1, 3)
    lines = np.arange(len(line_points), dtype=np.int32).reshape(-1, 2)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.broadcast_to(np.array([0.0, 0.95, 1.0]), (len(lines), 3))
    )
    return line_set


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively inspect complete pouring contact heat and GraspNet "
            "poses without rerunning either model."
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0/contact_propagation.npz"
        ),
    )
    parser.add_argument(
        "--grasps",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0/graspnet_selected.npy"
        ),
    )
    parser.add_argument(
        "--heat",
        choices=tuple(HEAT_KEYS),
        default="full",
        help="Initial heat field; keys 1-4 switch fields in the UI.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--show-pairs", type=int, default=40)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument(
        "--screenshot",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0/open3d_interactive_capture.png"
        ),
    )
    args = parser.parse_args()

    propagation = np.load(args.input)
    points = np.asarray(propagation["full_points_camera"], dtype=np.float32)
    heat_fields = {
        label: np.asarray(propagation[key], dtype=np.float32)
        for label, key in HEAT_KEYS.items()
    }
    grasps = np.load(args.grasps)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(_heat_colors(heat_fields[args.heat]))
    geometries: list[o3d.geometry.Geometry] = [
        cloud,
        *_grasp_geometries(grasps, args.top_k),
    ]
    pair_lines = _pair_lines(
        points,
        np.asarray(propagation["antipodal_pairs"]),
        args.show_pairs,
    )
    if pair_lines is not None:
        geometries.append(pair_lines)

    screenshot_path = Path(args.screenshot)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(
        window_name="LFV complete contact + GraspNet (Q/ESC close, S save)",
        width=1100,
        height=850,
    )
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    render = visualizer.get_render_option()
    render.background_color = np.array([0.97, 0.97, 0.97])
    render.point_size = args.point_size

    center = 0.5 * (
        np.quantile(points, 0.005, axis=0)
        + np.quantile(points, 0.995, axis=0)
    )
    view = visualizer.get_view_control()
    view.set_lookat(center.astype(float))
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_zoom(0.72)

    def set_heat(label: str):
        def callback(vis):
            cloud.colors = o3d.utility.Vector3dVector(
                _heat_colors(heat_fields[label])
            )
            vis.update_geometry(cloud)
            print(f"heat={label}")
            return False

        return callback

    def save_screenshot(vis):
        vis.capture_screen_image(str(screenshot_path), do_render=True)
        print(f"saved={screenshot_path}")
        return False

    visualizer.register_key_callback(ord("1"), set_heat("full"))
    visualizer.register_key_callback(ord("2"), set_heat("visible"))
    visualizer.register_key_callback(ord("3"), set_heat("pair_visible"))
    visualizer.register_key_callback(ord("4"), set_heat("opposite"))
    visualizer.register_key_callback(ord("S"), save_screenshot)
    print(
        "Mouse: left rotate, wheel zoom, Shift+left pan | "
        "1 full, 2 visible, 3 pair-visible, 4 opposite | S save | Q/ESC close"
    )
    visualizer.run()
    visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
