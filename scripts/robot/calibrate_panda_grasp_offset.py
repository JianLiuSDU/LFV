#!/usr/bin/env python3
"""Fast state-only search for a Panda TCP correction that can lift the mug."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.robot.panda_grasp_execution import (
    graspnet_object_row_to_panda_tcp_world,
    interpolate_se3,
    maniskill_wxyz_pose_to_matrix,
    pregrasp_pose,
    tcp_world_to_absolute_action,
)
from lfv.robot.gripper_extension import DEFAULT_LONG_FINGER_SPEC
from lfv_sim.maniskill.env_factory import make_env
from lfv_sim.maniskill.specs import get_task_spec


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _pose_matrix(pose_struct) -> np.ndarray:
    return maniskill_wxyz_pose_to_matrix(_numpy(pose_struct.raw_pose))


def _bool(value) -> bool:
    return bool(np.asarray(_numpy(value)).reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-report", required=True)
    parser.add_argument("--grasp-object", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--robot-uid",
        choices=("panda", "panda_long_finger"),
        default="panda",
    )
    parser.add_argument(
        "--orthogonal-offsets",
        type=float,
        nargs="+",
        default=[-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015],
    )
    parser.add_argument(
        "--closing-offsets",
        type=float,
        nargs="+",
        default=[-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015],
    )
    parser.add_argument("--approach-offset", type=float, default=0.0)
    args = parser.parse_args()

    snapshot_report = json.loads(
        Path(args.snapshot_report).expanduser().read_text(encoding="utf-8")
    )
    grasp_row = np.load(Path(args.grasp_object).expanduser()).astype(np.float32)
    spec = get_task_spec("pouring")
    env = make_env(
        spec,
        robot_uids=args.robot_uid,
        obs_mode="state",
        control_mode="pd_ee_pose",
        render_mode="rgb_array",
        extra_env_kwargs={
            "cup_asset": snapshot_report["cup_asset"],
            "max_episode_steps": 1000,
        },
    )
    records = []
    try:
        for orthogonal, closing in itertools.product(
            args.orthogonal_offsets, args.closing_offsets
        ):
            env.reset(seed=args.seed, options={"layout": snapshot_report["layout"]})
            unwrapped = env.unwrapped
            object_initial = _pose_matrix(unwrapped.cup.pose)
            tcp_initial = _pose_matrix(unwrapped.agent.tcp.pose)
            root_world = _pose_matrix(unwrapped.agent.robot.pose)
            tcp_grasp = graspnet_object_row_to_panda_tcp_world(grasp_row, object_initial)
            offset_local = np.asarray(
                [orthogonal, closing, args.approach_offset], dtype=np.float32
            )
            tcp_grasp[:3, 3] += tcp_grasp[:3, :3] @ offset_local
            tcp_pregrasp = pregrasp_pose(tcp_grasp, 0.08)

            def step_pose(pose, gripper, repeats=1):
                action = tcp_world_to_absolute_action(pose, root_world, gripper)
                info = None
                for _ in range(repeats):
                    _, _, _, _, info = env.step(action)
                return info

            for pose in interpolate_se3(tcp_initial, tcp_pregrasp, 20):
                step_pose(pose, 1.0, 2)
            for pose in interpolate_se3(tcp_pregrasp, tcp_grasp, 12):
                step_pose(pose, 1.0, 2)
            info = None
            for _ in range(35):
                info = step_pose(tcp_grasp, -1.0)
            cup_after_close = _pose_matrix(unwrapped.cup.pose)
            tcp_actual = _pose_matrix(unwrapped.agent.tcp.pose)
            qpos_after_close = _numpy(unwrapped.agent.robot.get_qpos())[-2:]
            grasped_after_close = _bool(info["is_grasped"])

            tcp_lift = tcp_grasp.copy()
            tcp_lift[2, 3] += 0.06
            max_cup_z = float(cup_after_close[2, 3])
            for pose in interpolate_se3(tcp_grasp, tcp_lift, 15):
                info = step_pose(pose, -1.0, 2)
                max_cup_z = max(max_cup_z, float(_pose_matrix(unwrapped.cup.pose)[2, 3]))
            cup_final = _pose_matrix(unwrapped.cup.pose)
            lift_delta = float(cup_final[2, 3] - object_initial[2, 3])
            max_lift_delta = float(max_cup_z - object_initial[2, 3])
            grasped_after_lift = _bool(info["is_grasped"])
            tcp_error_local = tcp_grasp[:3, :3].T @ (
                tcp_actual[:3, 3] - tcp_grasp[:3, 3]
            )
            score = (
                100.0 * float(grasped_after_lift)
                + 20.0 * float(grasped_after_close)
                + 10.0 * max(0.0, max_lift_delta)
                + float(np.mean(qpos_after_close))
            )
            record = {
                "offset_local_m": offset_local.astype(float).tolist(),
                "grasped_after_close": grasped_after_close,
                "grasped_after_lift": grasped_after_lift,
                "finger_qpos_after_close_m": qpos_after_close.astype(float).tolist(),
                "cup_final_lift_delta_m": lift_delta,
                "cup_max_lift_delta_m": max_lift_delta,
                "tcp_error_local_after_close_m": tcp_error_local.astype(float).tolist(),
                "score": score,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        env.close()

    records.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "robot_uid": args.robot_uid,
        "gripper_extension": (
            DEFAULT_LONG_FINGER_SPEC.to_dict()
            if args.robot_uid == "panda_long_finger"
            else None
        ),
        "coordinate_order": ["panda_tcp_orthogonal", "closing", "approach"],
        "num_trials": len(records),
        "selected": records[0],
        "successful_trials": [
            row for row in records if row["grasped_after_lift"] or row["cup_max_lift_delta_m"] > 0.025
        ],
        "trials": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SELECTED", json.dumps(records[0], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
