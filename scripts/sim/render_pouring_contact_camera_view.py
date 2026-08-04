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
    return (1.0 - alpha) * 0.62 + alpha * turbo_rgb


def _project_camera(
    points: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-8)
    return np.stack(
        (
            intrinsic[0, 0] * points[:, 0] / z + intrinsic[0, 2],
            intrinsic[1, 1] * points[:, 1] / z + intrinsic[1, 2],
        ),
        axis=-1,
    )


def _camera_crop(
    image: np.ndarray,
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    *,
    output_size: int,
    padding_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    pixels = _project_camera(points_camera, intrinsic)
    finite = (
        np.isfinite(pixels).all(axis=1)
        & (points_camera[:, 2] > 1e-5)
    )
    pixels = pixels[finite]
    lower = np.quantile(pixels, 0.002, axis=0)
    upper = np.quantile(pixels, 0.998, axis=0)
    extent = np.maximum(upper - lower, 1.0)
    padding = padding_ratio * float(max(extent))
    x0 = max(0, int(np.floor(lower[0] - padding)))
    y0 = max(0, int(np.floor(lower[1] - padding)))
    x1 = min(width, int(np.ceil(upper[0] + padding)) + 1)
    y1 = min(height, int(np.ceil(upper[1] + padding)) + 1)
    crop = image[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    square = np.full((side, side, 3), 247, dtype=np.uint8)
    offset_y = (side - crop.shape[0]) // 2
    offset_x = (side - crop.shape[1]) // 2
    square[
        offset_y : offset_y + crop.shape[0],
        offset_x : offset_x + crop.shape[1],
    ] = crop
    closeup = cv2.resize(
        square,
        (output_size, output_size),
        interpolation=(
            cv2.INTER_AREA if side > output_size else cv2.INTER_CUBIC
        ),
    )
    return closeup, (x0, y0, x1, y1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render complete pouring contact heat in the exact ManiSkill "
            "base_camera pinhole view, without grasps or 3-D view changes."
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
        "--snapshot",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0/pouring_snapshot.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/"
            "seed_0"
        ),
    )
    parser.add_argument("--heat", choices=tuple(HEAT_KEYS), default="full")
    parser.add_argument("--point-size", type=float, default=9.0)
    parser.add_argument(
        "--render-scale",
        type=int,
        default=4,
        help="Supersampling factor; intrinsic and image size are scaled together.",
    )
    parser.add_argument("--closeup-size", type=int, default=800)
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    args = parser.parse_args()

    propagation = np.load(args.input)
    snapshot = np.load(args.snapshot)
    points = np.asarray(propagation["full_points_camera"], dtype=np.float64)
    heat = np.asarray(propagation[HEAT_KEYS[args.heat]], dtype=np.float32)
    base_intrinsic = np.asarray(snapshot["intrinsic_cv"], dtype=np.float64)
    rgb = np.asarray(snapshot["rgb"], dtype=np.uint8)
    base_height, base_width = rgb.shape[:2]
    render_scale = max(1, args.render_scale)
    height = base_height * render_scale
    width = base_width * render_scale
    intrinsic = base_intrinsic.copy()
    intrinsic[0, :] *= render_scale
    intrinsic[1, :] *= render_scale
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"contact_{args.heat}_camera_view.png"
    closeup_path = output_dir / f"contact_{args.heat}_camera_closeup.png"

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(_heat_colors(heat))
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=width, height=height, visible=False)
    visualizer.add_geometry(cloud)
    render = visualizer.get_render_option()
    render.background_color = np.array([0.97, 0.97, 0.97])
    render.point_size = args.point_size

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
    )
    # The stored points are already in the base_camera OpenCV frame.
    camera.extrinsic = np.eye(4, dtype=np.float64)
    visualizer.get_view_control().convert_from_pinhole_camera_parameters(
        camera,
        allow_arbitrary=True,
    )
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(str(full_path), do_render=True)
    visualizer.destroy_window()

    full_bgr = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
    if full_bgr is None:
        raise RuntimeError(f"Open3D did not create {full_path}")
    closeup, crop_xyxy = _camera_crop(
        full_bgr,
        points,
        intrinsic,
        output_size=args.closeup_size,
        padding_ratio=args.padding_ratio,
    )
    cv2.imwrite(str(closeup_path), closeup)
    print(f"heat={args.heat}")
    print(f"camera_view={full_path}")
    print(f"camera_closeup={closeup_path}")
    print(f"crop_xyxy={crop_xyxy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
