#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np
import open3d as o3d


def grasp_points_from_T(T: np.ndarray, *, width: float, depth: float = 0.045) -> np.ndarray:
    """Local gripper keypoints in grasp frame: [lb, lt, rb, rt, tcp, approach_pt]."""
    depth_base = 0.02
    local = np.asarray(
        [
            [-depth_base, -width / 2, 0.0],  # 0: left back (finger base)
            [depth, -width / 2, 0.0],        # 1: left tip
            [-depth_base, width / 2, 0.0],   # 2: right back
            [depth, width / 2, 0.0],         # 3: right tip
            [0.0, 0.0, 0.0],                 # 4: TCP
            [depth, 0.0, 0.0],               # 5: approach direction point
        ],
        dtype=np.float64,
    )
    return (T[:3, :3] @ local.T).T + T[:3, 3][None]


def create_gripper_lines(T: np.ndarray, width: float, *, color=(1.0, 0.0, 0.0), approach_color=(0.0, 0.3, 1.0)):
    pts = grasp_points_from_T(T, width=width)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(
            np.asarray([[0, 1], [2, 3], [0, 2], [4, 5]], dtype=np.int32)
        ),
    )
    colors = np.asarray([color, color, color, approach_color], dtype=np.float64)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_contact_sphere(center: np.ndarray, color: list) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def print_diagnostics(
    label: np.lib.npyio.NpzFile,
    heat_data: np.lib.npyio.NpzFile,
    meta: dict | None,
    frame: str,
) -> None:
    valid = bool(label["valid"])
    quality = meta.get("quality", "?") if meta else "?"
    conf = float(label["confidence"])
    width = float(label["width_m"])
    selected_frame = int(label["selected_frame"])
    print(f"\n--- {frame} ---")
    print(f"  quality: {quality},  valid: {valid},  confidence: {conf:.3f}")
    print(f"  selected_frame: {selected_frame},  width_m: {width:.4f}")
    if not valid or np.isnan(width):
        print("  (reject / no valid grasp)")
        return

    T_cam = label["T_grasp_cam"].astype(np.float64)
    approach, closing = T_cam[:3, 0], T_cam[:3, 1]
    tcp = T_cam[:3, 3]
    q_thumb = label["q_thumb_cam"].astype(np.float64)
    q_index = label["q_index_cam"].astype(np.float64)
    center = 0.5 * (q_thumb + q_index)
    depth_into = float(np.dot(center - tcp, approach))
    print(f"  TCP          : {np.round(tcp, 4)}")
    print(f"  approach     : {np.round(approach, 4)}")
    print(f"  closing      : {np.round(closing, 4)}")
    print(f"  center       : {np.round(center, 4)}")
    print(f"  depth_into_gripper (should be ~0.045): {depth_into:.4f}")
    print(f"  thumb_to_surface dist: {float(np.linalg.norm(q_thumb - center)):.4f}  (expected ~0)")
    print(f"  index_to_surface dist: {float(np.linalg.norm(q_index - center)):.4f}  (expected ~0)")

    # Heat at contact points
    if "pixels_uv" in heat_data and "contact_heat" in heat_data:
        pixels = heat_data["pixels_uv"].astype(np.int64)
        heat = heat_data["contact_heat"].astype(np.float64)
        proj = {}
        for name, q in [("thumb", q_thumb), ("index", q_index)]:
            dist = np.linalg.norm(heat_data["points_camera"] - q, axis=1)
            nearest = int(np.argmin(dist))
            proj[name] = float(heat[nearest])
        print(f"  heat@nearest_point thumb={proj['thumb']:.3f} index={proj['index']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open3D viewer for HaMeR thumb-index grasp pseudo label.")
    parser.add_argument("--episode-dir", required=True, help="Path to one episode directory.")
    parser.add_argument("--label-dir", default=None, help="Defaults to <episode-dir>/hamer_grasp_pseudo_label.")
    parser.add_argument("--object-frame", action="store_true", help="Show in grasp_pseudo_label object frame (camera axes, object-center origin).")
    parser.add_argument("--heat-field", default="contact_heat", choices=["contact_heat", "contact_heat_raw"], help="Heat field to color the point cloud.")
    parser.add_argument("--show-candidates", action="store_true", help="Also draw all window candidates in light gray.")
    parser.add_argument("--no-heat", action="store_true", help="Color point cloud by z instead of heat.")
    args = parser.parse_args()

    ep = pathlib.Path(args.episode_dir)
    label_dir = pathlib.Path(args.label_dir) if args.label_dir else ep / "hamer_grasp_pseudo_label"
    heat_path = ep / "contact_heatmap" / "contact_heatmap.npz"
    label_path = label_dir / "grasp_pseudo_label.npz"
    meta_path = label_dir / "grasp_pseudo_label_meta.json"

    if not heat_path.exists():
        print(f"[viewer] missing contact heatmap: {heat_path}")
        return 1
    if not label_path.exists():
        print(f"[viewer] missing grasp label: {label_path}")
        return 1

    heat_data = np.load(heat_path)
    label = np.load(label_path)
    meta = json.load(open(meta_path)) if meta_path.exists() else None

    suffix = "object" if args.object_frame else "cam"
    points = heat_data[f"points_{suffix}_m" if args.object_frame else "points_camera"].astype(np.float64)
    heat = heat_data[args.heat_field].astype(np.float64)
    q_thumb = label[f"q_thumb_{suffix}"].astype(np.float64)
    q_index = label[f"q_index_{suffix}"].astype(np.float64)
    T = label[f"T_grasp_{suffix}"].astype(np.float64)
    width = float(label["width_m"])

    print_diagnostics(label, heat_data, meta, ep.name)

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if args.no_heat:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
    else:
        colors = mpl.colormaps["magma"](np.clip(heat, 0, 1))[:, :3].astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries: list = [pcd]

    # Contact points
    geometries.append(create_contact_sphere(q_thumb, [1.0, 0.0, 1.0]))  # magenta = thumb
    geometries.append(create_contact_sphere(q_index, [0.0, 1.0, 0.0]))  # lime = index

    # Gripper (skip for reject)
    valid = bool(label["valid"])
    if valid and not np.isnan(width):
        geometries.append(create_gripper_lines(T, width))
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=T[:3, 3]))

        if args.show_candidates and f"candidate_T_{suffix}" in label:
            cand_T = label[f"candidate_T_{suffix}"].astype(np.float64)
            cand_w = label["candidate_width_m"].astype(np.float64)
            for i in range(len(cand_T)):
                geometries.append(create_gripper_lines(cand_T[i], float(cand_w[i]), color=(0.5, 0.5, 0.5), approach_color=(0.4, 0.4, 0.6)))
    else:
        print("[viewer] no gripper to show (reject or invalid)")

    frame_title = "object" if args.object_frame else "camera"
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{ep.name} HaMeR thumb-index grasp ({frame_title} frame)",
        width=1280,
        height=960,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
