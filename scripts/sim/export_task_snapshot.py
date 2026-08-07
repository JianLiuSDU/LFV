#!/usr/bin/env python3
"""Export a task-neutral LFV RGB-D/segmentation/complete-surface snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from transforms3d.quaternions import quat2mat
import trimesh
from mani_skill.utils.geometry.trimesh_utils import get_render_shape_meshes, merge_meshes


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


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


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
        raise ValueError(f"Expected [x,y,z,qw,qx,qy,qz], got {raw_pose.shape}")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = quat2mat(raw_pose[3:]).astype(np.float32)
    transform[:3, 3] = raw_pose[:3]
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points, dtype=np.float32) @ transform[:3, :3].T
        + transform[:3, 3]
    ).astype(np.float32)


def _entity(env, attr: str | None):
    if not attr:
        raise ValueError("Task spec does not define an entity attribute")
    return getattr(env.unwrapped, attr)


def _surface_sample(entity, count: int, seed: int):
    if hasattr(entity, "get_first_collision_mesh"):
        mesh = entity.get_first_collision_mesh(to_world_frame=False)
    elif hasattr(entity, "render_shapes"):
        render_meshes = []
        shape_batches = entity.render_shapes
        if shape_batches:
            for render_shape in shape_batches[0]:
                render_meshes.extend(get_render_shape_meshes(render_shape))
        mesh = merge_meshes(render_meshes) if render_meshes else None
    else:
        mesh = None
    if mesh is None:
        raise RuntimeError(f"Entity {getattr(entity, 'name', entity)} has no collision mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            process=False,
        )
    points, face_ids = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return (
        mesh,
        np.asarray(points, dtype=np.float32),
        np.asarray(mesh.face_normals[face_ids], dtype=np.float32),
    )


def _overlay(rgb: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]]):
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    layer = canvas.copy()
    for mask, color in masks:
        layer[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(canvas, 0.68, layer, 0.32, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("pouring", "drawer_open", "picknplace"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--num-manipulated-points", type=int, default=256)
    parser.add_argument("--num-reference-points", type=int, default=256)
    parser.add_argument("--num-full-manipulated-points", type=int, default=30000)
    parser.add_argument("--num-complete-scene-points", type=int, default=60000)
    parser.add_argument("--drawer-x", type=float, default=-0.08)
    parser.add_argument("--drawer-y", type=float, default=0.02)
    parser.add_argument("--drawer-yaw", type=float, default=0.0)
    parser.add_argument("--initial-open", type=float, default=0.0)
    parser.add_argument("--banana-x", type=float, default=-0.10)
    parser.add_argument("--banana-y", type=float, default=-0.20)
    parser.add_argument("--banana-yaw", type=float, default=float(np.pi / 2))
    parser.add_argument("--plate-x", type=float, default=-0.10)
    parser.add_argument("--plate-y", type=float, default=0.20)
    parser.add_argument("--plate-yaw", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = get_task_spec(args.task)
    layout = {}
    if args.task == "drawer_open":
        layout = {
            "drawer_xy": [args.drawer_x, args.drawer_y],
            "drawer_yaw": args.drawer_yaw,
            "initial_open": args.initial_open,
        }
    elif args.task == "picknplace":
        layout = {
            "banana_xy": [args.banana_x, args.banana_y],
            "banana_yaw": args.banana_yaw,
            "plate_xy": [args.plate_x, args.plate_y],
            "plate_yaw": args.plate_yaw,
        }
    rng = np.random.default_rng(args.seed)

    env = make_env(spec, shader=args.shader)
    try:
        obs, _ = env.reset(seed=args.seed, options={"layout": layout})
        camera = extract_camera_observation(obs, spec.camera_uid)
        manipulated_ids = find_segmentation_ids(env, spec.manipulated_query)
        reference_ids = find_segmentation_ids(env, spec.target_query or "")
        object_ids = find_segmentation_ids(env, spec.object_query or spec.manipulated_query)
        manipulated_mask = object_mask(camera, manipulated_ids)
        reference_mask = object_mask(camera, reference_ids)
        foreground_mask = object_mask(camera, object_ids)
        if not manipulated_mask.any() or not reference_mask.any() or not foreground_mask.any():
            raise RuntimeError(
                "Empty task mask: "
                f"manipulated={manipulated_mask.sum()} reference={reference_mask.sum()} "
                f"foreground={foreground_mask.sum()} ids={(manipulated_ids, reference_ids, object_ids)}"
            )

        depth_m = normalize_depth(camera.depth)
        manipulated_pixels = uniform_grid_sample_mask_pixels(
            manipulated_mask, args.num_manipulated_points, rng=rng
        )
        reference_pixels = uniform_grid_sample_mask_pixels(
            reference_mask, args.num_reference_points, rng=rng
        )
        manipulated_points_camera = pixels_to_points_camera(
            depth_m, camera.intrinsic_cv, manipulated_pixels
        )
        reference_points_camera = pixels_to_points_camera(
            depth_m, camera.intrinsic_cv, reference_pixels
        )

        T_world_to_camera = _matrix4(camera.extrinsic_cv)
        manipulated_entity = _entity(env, spec.manipulated_entity_attr)
        reference_entity = _entity(env, spec.reference_entity_attr)
        T_manipulated_to_world = _pose_matrix(_numpy(manipulated_entity.pose.raw_pose))
        T_reference_to_world = _pose_matrix(_numpy(reference_entity.pose.raw_pose))
        T_manipulated_to_camera = T_world_to_camera @ T_manipulated_to_world
        T_camera_to_manipulated = np.linalg.inv(T_manipulated_to_camera).astype(np.float32)
        manipulated_points_local = _transform_points(
            manipulated_points_camera, T_camera_to_manipulated
        )

        mesh, full_points_local, full_normals_local = _surface_sample(
            manipulated_entity, args.num_full_manipulated_points, args.seed
        )
        full_points_camera = _transform_points(full_points_local, T_manipulated_to_camera)
        full_normals_camera = (
            full_normals_local @ T_manipulated_to_camera[:3, :3].T
        ).astype(np.float32)

        complete_points = []
        complete_normals = []
        complete_labels = []
        complete_colors = []
        complete_is_manipulated = []
        if args.task == "drawer_open":
            entities = [
                (env.unwrapped.cabinet_link, False, [0.08, 0.38, 0.27]),
                (env.unwrapped.moving_link, False, [0.06, 0.29, 0.21]),
                (env.unwrapped.handle_link, True, [0.02, 0.02, 0.02]),
            ]
        else:
            entities = [
                (manipulated_entity, True, [0.10, 0.20, 0.70]),
                (reference_entity, False, [0.65, 0.65, 0.65]),
            ]
        per_entity = max(1000, args.num_complete_scene_points // len(entities))
        for label, (entity, is_manipulated, color) in enumerate(entities):
            _, points_local, normals_local = _surface_sample(
                entity, per_entity, args.seed + 101 * (label + 1)
            )
            T_entity_to_world = _pose_matrix(_numpy(entity.pose.raw_pose))
            T_entity_to_camera = T_world_to_camera @ T_entity_to_world
            complete_points.append(_transform_points(points_local, T_entity_to_camera))
            complete_normals.append(normals_local @ T_entity_to_camera[:3, :3].T)
            complete_labels.append(np.full(len(points_local), label, dtype=np.int16))
            complete_colors.append(
                np.broadcast_to(np.asarray(color, dtype=np.float32), (len(points_local), 3))
            )
            complete_is_manipulated.append(
                np.full(len(points_local), is_manipulated, dtype=bool)
            )
        complete_points_camera = np.concatenate(complete_points).astype(np.float32)
        complete_normals_camera = np.concatenate(complete_normals).astype(np.float32)
        complete_scene_labels = np.concatenate(complete_labels)
        complete_scene_colors = np.concatenate(complete_colors).astype(np.float32)
        complete_scene_is_manipulated = np.concatenate(complete_is_manipulated)

        dense_points = depth_to_points_camera(depth_m, camera.intrinsic_cv)
        valid_depth = np.isfinite(depth_m) & (depth_m > 1e-4) & (depth_m < 2.0)
        scene_points_camera = dense_points[valid_depth].astype(np.float32)
        scene_colors = camera.rgb[valid_depth].astype(np.uint8)

        pull_axis_world = np.zeros(3, dtype=np.float32)
        if args.task == "drawer_open":
            pull_axis_world = (T_reference_to_world[:3, :3] @ np.asarray([1, 0, 0])).astype(np.float32)

        snapshot_path = output_dir / "task_snapshot.npz"
        np.savez_compressed(
            snapshot_path,
            task=np.asarray(args.task),
            rgb=camera.rgb,
            depth_m=depth_m,
            intrinsic_cv=camera.intrinsic_cv,
            T_world_to_camera=T_world_to_camera,
            T_manipulated_to_world=T_manipulated_to_world,
            T_reference_to_world=T_reference_to_world,
            T_manipulated_to_camera=T_manipulated_to_camera,
            object_mask=foreground_mask.astype(np.uint8),
            manipulated_mask=manipulated_mask.astype(np.uint8),
            reference_mask=reference_mask.astype(np.uint8),
            manipulated_pixels_uv=manipulated_pixels,
            manipulated_points_camera=manipulated_points_camera,
            manipulated_points_local=manipulated_points_local,
            manipulated_colors=camera.rgb[manipulated_pixels[:, 1], manipulated_pixels[:, 0]],
            reference_pixels_uv=reference_pixels,
            reference_points_camera=reference_points_camera,
            reference_colors=camera.rgb[reference_pixels[:, 1], reference_pixels[:, 0]],
            full_manipulated_points_local=full_points_local,
            full_manipulated_normals_local=full_normals_local,
            full_manipulated_points_camera=full_points_camera,
            full_manipulated_normals_camera=full_normals_camera,
            manipulated_mesh_vertices_local=np.asarray(mesh.vertices, dtype=np.float32),
            manipulated_mesh_faces=np.asarray(mesh.faces, dtype=np.int32),
            complete_scene_points_camera=complete_points_camera,
            complete_scene_normals_camera=complete_normals_camera,
            complete_scene_labels=complete_scene_labels,
            complete_scene_colors=complete_scene_colors,
            complete_scene_is_manipulated=complete_scene_is_manipulated,
            scene_points_camera=scene_points_camera,
            scene_colors=scene_colors,
            scene_is_manipulated=manipulated_mask[valid_depth].astype(bool),
            pull_axis_world=pull_axis_world,
        )

        cv2.imwrite(str(output_dir / "rgb_base_camera.png"), cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / "object_mask.png"), foreground_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "manipulated_mask.png"), manipulated_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "reference_mask.png"), reference_mask.astype(np.uint8) * 255)
        cv2.imwrite(
            str(output_dir / "semantic_overlay.png"),
            _overlay(camera.rgb, [(reference_mask, (55, 190, 55)), (manipulated_mask, (20, 50, 255))]),
        )
        cv2.imwrite(
            str(output_dir / "depth_mm.png"),
            np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16),
        )

        report = {
            "task": args.task,
            "seed": args.seed,
            "layout": layout,
            "camera": spec.camera_uid,
            "instruction": spec.instruction,
            "segmentation_ids": {
                "object": object_ids,
                "manipulated": manipulated_ids,
                "reference": reference_ids,
            },
            "counts": {
                "object_mask_pixels": int(foreground_mask.sum()),
                "manipulated_mask_pixels": int(manipulated_mask.sum()),
                "reference_mask_pixels": int(reference_mask.sum()),
                "full_manipulated_points": int(len(full_points_local)),
                "complete_scene_points": int(len(complete_points_camera)),
            },
            "pull_axis_world": pull_axis_world.tolist(),
            "coordinates": {
                "camera": "OpenCV: +X right, +Y down, +Z forward",
                "manipulated_local": "rigid manipulated entity/link frame",
                "pull_axis_world": "positive drawer-opening direction",
            },
            "snapshot": str(snapshot_path),
        }
        (output_dir / "snapshot_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
