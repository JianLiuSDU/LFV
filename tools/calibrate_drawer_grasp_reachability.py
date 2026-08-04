#!/usr/bin/env python3
"""State-only search for a reachable LFV drawer placement.

The GraspNet row stays in the manipulated-link frame.  Each trial changes only
the task layout, transforms exactly the same grasp into world coordinates, and
executes the normal pregrasp/approach/full-close sequence.  This separates arm
reachability from contact-transfer and grasp-ranking quality.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.robot.panda_grasp_execution import (
    graspnet_object_row_to_panda_tcp_world,
    interpolate_se3,
    maniskill_wxyz_pose_to_matrix,
    pregrasp_pose,
    tcp_world_to_absolute_action,
)
from lfv.robot.gripper_extension import (
    DEFAULT_LONG_FINGER_SPEC,
    DRAWER_LONG_FINGER_SPEC,
)
from lfv_sim.maniskill.env_factory import make_env
from lfv_sim.maniskill.specs import get_task_spec


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim and value.shape[0] == 1:
        value = value[0]
    return value


def _matrix(pose) -> np.ndarray:
    return maniskill_wxyz_pose_to_matrix(_numpy(pose.raw_pose))


def _bool(value) -> bool:
    return bool(np.asarray(_numpy(value)).reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp-object", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drawer-xs", type=float, nargs="+", default=[-0.10, -0.14, -0.18, -0.22])
    parser.add_argument("--drawer-ys", type=float, nargs="+", default=[-0.02, 0.0, 0.02])
    parser.add_argument("--drawer-zs", type=float, nargs="+", default=[0.004])
    parser.add_argument(
        "--approach-offsets",
        type=float,
        nargs="+",
        default=[0.0],
        help="Offsets along Panda TCP local +Z / GraspNet approach axis.",
    )
    parser.add_argument("--drawer-yaw", type=float, default=float(np.pi))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--robot-uid",
        choices=("panda", "panda_long_finger", "panda_drawer_finger"),
        default="panda_drawer_finger",
    )
    parser.add_argument("--pregrasp-distance", type=float, default=0.08)
    parser.add_argument(
        "--approach-gripper-action",
        type=float,
        default=1.0,
        help=(
            "Normalized Panda gripper command used while moving to and "
            "descending from pregrasp; -1 is closed and +1 is fully open."
        ),
    )
    parser.add_argument("--move-waypoints", type=int, default=28)
    parser.add_argument("--approach-waypoints", type=int, default=18)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--close-steps", type=int, default=45)
    args = parser.parse_args()

    grasp = np.load(Path(args.grasp_object).expanduser()).astype(np.float32)
    spec = get_task_spec("drawer_open")
    env = make_env(
        spec,
        robot_uids=args.robot_uid,
        obs_mode="state",
        control_mode="pd_ee_pose",
        extra_env_kwargs={"max_episode_steps": 1000},
    )
    records = []
    try:
        for drawer_x, drawer_y, drawer_z, approach_offset in itertools.product(
            args.drawer_xs, args.drawer_ys, args.drawer_zs, args.approach_offsets
        ):
            layout = {
                "drawer_xy": [drawer_x, drawer_y],
                "drawer_z": drawer_z,
                "drawer_yaw": args.drawer_yaw,
                "initial_open": 0.0,
            }
            env.reset(seed=args.seed, options={"layout": layout})
            unwrapped = env.unwrapped
            manipulated = _matrix(unwrapped.handle_link.pose)
            initial_tcp = _matrix(unwrapped.agent.tcp.pose)
            root = _matrix(unwrapped.agent.robot.pose)
            target = graspnet_object_row_to_panda_tcp_world(grasp, manipulated)
            target[:3, 3] += target[:3, 2] * float(approach_offset)
            pregrasp = pregrasp_pose(target, args.pregrasp_distance)

            def step(pose, gripper, repeats=1):
                info = None
                action = tcp_world_to_absolute_action(pose, root, gripper)
                for _ in range(repeats):
                    _, _, _, _, info = env.step(action)
                return info

            for pose in interpolate_se3(initial_tcp, pregrasp, args.move_waypoints):
                step(pose, args.approach_gripper_action, args.settle_steps)
            for pose in interpolate_se3(pregrasp, target, args.approach_waypoints):
                step(pose, args.approach_gripper_action, args.settle_steps)
            qpos_before = _numpy(unwrapped.agent.robot.get_qpos())[-2:]
            plate_spec = (
                DRAWER_LONG_FINGER_SPEC
                if args.robot_uid == "panda_drawer_finger"
                else DEFAULT_LONG_FINGER_SPEC
                if args.robot_uid == "panda_long_finger"
                else None
            )
            plate_geometry = {}
            if plate_spec is not None:
                for side, link_name in (
                    ("left", "panda_leftfinger"),
                    ("right", "panda_rightfinger"),
                ):
                    link = unwrapped.agent.robot.links_map[link_name]
                    T_link = _matrix(link.pose)
                    center_local = np.asarray(
                        plate_spec.center_for_side(side), dtype=np.float32
                    )
                    center_world = T_link[:3, :3] @ center_local + T_link[:3, 3]
                    plate_geometry[side] = {
                        "center_world_m": center_world.astype(float).tolist(),
                        "link_axes_world": T_link[:3, :3].astype(float).tolist(),
                        "half_size_local_m": list(plate_spec.half_size_m),
                    }
            info = None
            for _ in range(args.close_steps):
                info = step(target, -1.0)

            actual = _matrix(unwrapped.agent.tcp.pose)
            error_world = actual[:3, 3] - target[:3, 3]
            error_local = target[:3, :3].T @ error_world
            qpos = _numpy(unwrapped.agent.robot.get_qpos())[-2:]
            grasped = _bool(info["is_grasped"])
            record = {
                "layout": layout,
                "approach_offset_m": approach_offset,
                "grasped": grasped,
                "target_tcp_world_m": target[:3, 3].astype(float).tolist(),
                "actual_tcp_world_m": actual[:3, 3].astype(float).tolist(),
                "tcp_error_norm_m": float(np.linalg.norm(error_world)),
                "tcp_error_local_m": error_local.astype(float).tolist(),
                "approach_gripper_action": args.approach_gripper_action,
                "finger_qpos_before_close_m": qpos_before.astype(float).tolist(),
                "plate_geometry_before_close": plate_geometry,
                "finger_qpos_m": qpos.astype(float).tolist(),
                "drawer_qpos_m": float(_numpy(unwrapped.drawer_qpos)),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        env.close()

    records.sort(key=lambda row: (not row["grasped"], row["tcp_error_norm_m"]))
    report = {
        "purpose": "scene-placement reachability; the object-frame grasp is fixed",
        "robot_uid": args.robot_uid,
        "num_trials": len(records),
        "selected": records[0],
        "successful_trials": [row for row in records if row["grasped"]],
        "trials": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SELECTED", json.dumps(records[0], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
