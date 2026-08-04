#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from transforms3d.quaternions import quat2mat
import trimesh


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv_sim.maniskill.env_factory import make_env
from lfv_sim.maniskill.perception import (
    extract_camera_observation,
    find_segmentation_ids,
    object_mask,
)
from lfv_sim.maniskill.pointcloud import (
    depth_to_points_camera,
    normalize_depth,
    pixels_to_points_camera,
    uniform_grid_sample_mask_pixels,
)
from lfv_sim.maniskill.specs import get_task_spec


def _matrix4(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        output = np.eye(4, dtype=np.float32)
        output[:3] = matrix
        return output
    raise ValueError(f"Expected [3,4] or [4,4] transform, got {matrix.shape}")


def _pose_matrix(raw_pose: np.ndarray) -> np.ndarray:
    raw_pose = np.asarray(raw_pose, dtype=np.float32).reshape(-1)
    if raw_pose.shape != (7,):
        raise ValueError(f"Expected pose [x,y,z,qw,qx,qy,qz], got {raw_pose.shape}")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = quat2mat(raw_pose[3:7]).astype(np.float32)
    transform[:3, 3] = raw_pose[:3]
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points, dtype=np.float32) @ transform[:3, :3].T
        + transform[:3, 3]
    ).astype(np.float32)


def _sampling_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = np.asarray([0, 130, 255], dtype=np.uint8)
    overlay = canvas.copy()
    overlay[mask] = color
    canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0.0)
    for u, v in pixels:
        cv2.circle(canvas, (int(u), int(v)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export an LFV ManiSkill3 pouring RGB-D snapshot and complete mug geometry."
    )
    parser.add_argument(
        "--output-dir",
        default="/home/users1/ljian/lfv_runs/pouring_complete_grasp_validation/seed_0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-visible-points", type=int, default=128)
    parser.add_argument("--num-target-points", type=int, default=256)
    parser.add_argument("--num-full-points", type=int, default=30000)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--cup-asset-kind", choices=["ycb", "mesh"], default="ycb")
    parser.add_argument("--cup-model-id", default="025_mug")
    parser.add_argument("--cup-asset-id", default=None)
    parser.add_argument("--cup-visual-file", default=None)
    parser.add_argument("--cup-collision-glob", default=None)
    parser.add_argument("--cup-scale", type=float, default=1.0)
    parser.add_argument("--cup-density", type=float, default=1000.0)
    parser.add_argument(
        "--cup-base-color",
        type=float,
        nargs=4,
        default=[0.08, 0.18, 0.65, 1.0],
        metavar=("R", "G", "B", "A"),
    )
    parser.add_argument("--cup-x", type=float, default=-0.12)
    parser.add_argument("--cup-y", type=float, default=0.09)
    parser.add_argument("--cup-yaw", type=float, default=-np.pi / 2)
    parser.add_argument("--bowl-x", type=float, default=0.08)
    parser.add_argument("--bowl-y", type=float, default=-0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    layout = {
        "cup_xy": [args.cup_x, args.cup_y],
        "cup_yaw": args.cup_yaw,
        "bowl_xy": [args.bowl_x, args.bowl_y],
    }

    if args.cup_asset_kind == "mesh":
        if not args.cup_visual_file or not args.cup_collision_glob:
            parser.error(
                "mesh cups require --cup-visual-file and --cup-collision-glob"
            )
        collision_files = sorted(glob.glob(args.cup_collision_glob))
        if not collision_files:
            parser.error(
                f"--cup-collision-glob matched no files: {args.cup_collision_glob}"
            )
        cup_asset = {
            "kind": "mesh",
            "asset_id": args.cup_asset_id or Path(args.cup_visual_file).parent.name,
            "visual_file": str(Path(args.cup_visual_file).expanduser().resolve()),
            "collision_files": [
                str(Path(path).expanduser().resolve()) for path in collision_files
            ],
            "scale": args.cup_scale,
            "density": args.cup_density,
            "base_color": args.cup_base_color,
        }
    else:
        cup_asset = {
            "kind": "ycb",
            "asset_id": args.cup_asset_id or args.cup_model_id,
            "model_id": args.cup_model_id,
        }

    spec = get_task_spec("pouring")
    env = make_env(
        spec,
        shader=args.shader,
        extra_env_kwargs={"cup_asset": cup_asset},
    )
    try:
        obs, _ = env.reset(seed=args.seed, options={"layout": layout})
        camera = extract_camera_observation(obs, spec.camera_uid)
        cup_ids = find_segmentation_ids(env, spec.manipulated_query)
        cup_mask = object_mask(camera, cup_ids)
        bowl_ids = find_segmentation_ids(env, spec.target_query)
        bowl_mask = object_mask(camera, bowl_ids)
        depth_m = normalize_depth(camera.depth)
        pixels_uv = uniform_grid_sample_mask_pixels(
            cup_mask,
            args.num_visible_points,
            rng=rng,
        )
        visible_points_camera = pixels_to_points_camera(
            depth_m,
            camera.intrinsic_cv,
            pixels_uv,
        )
        target_pixels_uv = uniform_grid_sample_mask_pixels(
            bowl_mask,
            args.num_target_points,
            rng=rng,
        )
        target_points_camera = pixels_to_points_camera(
            depth_m,
            camera.intrinsic_cv,
            target_pixels_uv,
        )

        T_world_to_camera = _matrix4(camera.extrinsic_cv)
        cup_actor = env.unwrapped.cup
        cup_pose = cup_actor.pose.raw_pose.detach().cpu().numpy()[0]
        T_object_to_world = _pose_matrix(cup_pose)
        T_object_to_camera = T_world_to_camera @ T_object_to_world
        T_camera_to_object = np.linalg.inv(T_object_to_camera).astype(np.float32)
        visible_points_object = _transform_points(
            visible_points_camera,
            T_camera_to_object,
        )

        if cup_asset["kind"] == "mesh":
            mesh_object = trimesh.load(
                cup_asset["visual_file"],
                force="mesh",
                process=False,
            )
            mesh_object.apply_scale(float(cup_asset["scale"]))
            full_surface_source = "custom visual mesh"
        else:
            mesh_object = cup_actor.get_first_collision_mesh(to_world_frame=False)
            full_surface_source = "actor collision mesh"
        if mesh_object is None:
            raise RuntimeError("The ManiSkill cup actor did not expose a collision mesh")
        if not isinstance(mesh_object, trimesh.Trimesh):
            mesh_object = trimesh.Trimesh(
                vertices=np.asarray(mesh_object.vertices),
                faces=np.asarray(mesh_object.faces),
                process=False,
            )
        full_points_object, face_indices = trimesh.sample.sample_surface(
            mesh_object,
            args.num_full_points,
            seed=args.seed,
        )
        full_points_object = np.asarray(full_points_object, dtype=np.float32)
        full_normals_object = np.asarray(
            mesh_object.face_normals[face_indices],
            dtype=np.float32,
        )
        full_points_camera = _transform_points(
            full_points_object,
            T_object_to_camera,
        )
        full_normals_camera = (
            full_normals_object @ T_object_to_camera[:3, :3].T
        ).astype(np.float32)

        dense_points = depth_to_points_camera(depth_m, camera.intrinsic_cv)
        valid_depth = np.isfinite(depth_m) & (depth_m > 1e-4) & (depth_m < 2.0)
        scene_points_camera = dense_points[valid_depth].astype(np.float32)
        scene_colors = camera.rgb[valid_depth].astype(np.uint8)
        scene_is_cup = cup_mask[valid_depth].astype(bool)

        cv2.imwrite(
            str(output_dir / "rgb_base_camera.png"),
            cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(output_dir / "cup_mask.png"), cup_mask.astype(np.uint8) * 255)
        cv2.imwrite(
            str(output_dir / "bowl_mask.png"),
            bowl_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(output_dir / "depth_mm.png"),
            np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16),
        )
        cv2.imwrite(
            str(output_dir / "visible_sampling.png"),
            _sampling_overlay(camera.rgb, cup_mask, pixels_uv),
        )

        np.savez_compressed(
            output_dir / "pouring_snapshot.npz",
            rgb=camera.rgb,
            depth_m=depth_m,
            cup_mask=cup_mask.astype(np.uint8),
            bowl_mask=bowl_mask.astype(np.uint8),
            intrinsic_cv=camera.intrinsic_cv,
            T_world_to_camera=T_world_to_camera,
            T_object_to_world=T_object_to_world,
            T_object_to_camera=T_object_to_camera,
            visible_pixels_uv=pixels_uv,
            visible_points_camera=visible_points_camera,
            visible_points_object=visible_points_object,
            visible_colors=camera.rgb[pixels_uv[:, 1], pixels_uv[:, 0]],
            target_pixels_uv=target_pixels_uv,
            target_points_camera=target_points_camera,
            target_colors=camera.rgb[
                target_pixels_uv[:, 1], target_pixels_uv[:, 0]
            ],
            full_points_object=full_points_object,
            full_normals_object=full_normals_object,
            full_points_camera=full_points_camera,
            full_normals_camera=full_normals_camera,
            mesh_vertices_object=np.asarray(mesh_object.vertices, dtype=np.float32),
            mesh_faces=np.asarray(mesh_object.faces, dtype=np.int32),
            scene_points_camera=scene_points_camera,
            scene_colors=scene_colors,
            scene_is_cup=scene_is_cup,
        )
        report = {
            "task": "pouring",
            "seed": args.seed,
            "cup_asset": cup_asset,
            "layout": layout,
            "camera": spec.camera_uid,
            "cup_segmentation_ids": cup_ids,
            "bowl_segmentation_ids": bowl_ids,
            "visible_point_count": int(len(visible_points_object)),
            "target_point_count": int(len(target_points_camera)),
            "full_point_count": int(len(full_points_object)),
            "scene_point_count": int(len(scene_points_camera)),
            "mesh_vertices": int(len(mesh_object.vertices)),
            "mesh_faces": int(len(mesh_object.faces)),
            "mesh_watertight": bool(mesh_object.is_watertight),
            "full_surface_source": full_surface_source,
            "coordinates": {
                "visible_points_object": "cup actor local frame",
                "full_points_object": "cup actor local frame",
                "full_points_camera": "ManiSkill base_camera OpenCV frame",
                "target_points_camera": "ManiSkill base_camera OpenCV frame",
                "T_object_to_camera": "left-multiplies homogeneous object points",
            },
        }
        (output_dir / "snapshot_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
