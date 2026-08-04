#!/usr/bin/env python3
"""Execute a contact-constrained GraspNet pose and learned LFV motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.robot.gripper_extension import (
    DEFAULT_LONG_FINGER_SPEC,
    DRAWER_LONG_FINGER_SPEC,
)
from lfv.robot.panda_grasp_execution import (
    graspnet_object_row_to_panda_tcp_world,
    interpolate_se3,
    maniskill_wxyz_pose_to_matrix,
    object_poses_to_tcp_poses,
    pregrasp_pose,
    project_prismatic_trajectory,
    tcp_world_to_absolute_action,
)
from lfv_sim.maniskill.env_factory import make_env
from lfv_sim.maniskill.perception import extract_camera_observation
from lfv_sim.maniskill.specs import get_task_spec


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _bool(value) -> bool:
    return bool(np.asarray(_numpy(value)).reshape(-1)[0])


def _pose_matrix(pose_struct) -> np.ndarray:
    return maniskill_wxyz_pose_to_matrix(_numpy(pose_struct.raw_pose))


def _rgb(value) -> np.ndarray:
    frame = _numpy(value)
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255)
    return np.asarray(frame, dtype=np.uint8)


class Recorder:
    def __init__(self, env, output: Path, task: str, camera_uid: str, fps: int):
        self.env, self.output, self.task = env, output, task
        self.camera_uid, self.fps = camera_uid, fps
        self.oblique = self.front = None
        self.last_front = None
        self.frames = 0
        self.keyframes = {}

    @staticmethod
    def _writer(path: Path, fps: int, frame: np.ndarray):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (frame.shape[1], frame.shape[0]))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")
        return writer

    def _annotate(self, rgb: np.ndarray, phase: str, grasped: bool, view: str):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 68), (18, 18, 18), -1)
        cv2.putText(bgr, f"{self.task} | {view} | {phase}", (16, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"GraspNet + learned Full64 | grasped: {str(grasped).lower()}",
                    (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (80, 225, 255) if grasped else (160, 160, 255), 1, cv2.LINE_AA)
        return bgr

    def capture(self, phase: str, grasped: bool, obs=None, keyframe: str | None = None):
        oblique = self._annotate(_rgb(self.env.render()), phase, grasped, "oblique")
        if obs is not None:
            self.last_front = extract_camera_observation(obs, self.camera_uid).rgb
        if self.last_front is None:
            raise RuntimeError("Front-camera observation required")
        front = self._annotate(self.last_front, phase, grasped, "front")
        if self.oblique is None:
            self.oblique = self._writer(self.output / f"{self.task}_execution.mp4", self.fps, oblique)
            self.front = self._writer(self.output / f"{self.task}_execution_front.mp4", self.fps, front)
        self.oblique.write(oblique)
        self.front.write(front)
        self.frames += 1
        if keyframe:
            paths = []
            for prefix, frame in (("", oblique), ("front_", front)):
                path = self.output / f"keyframe_{prefix}{keyframe}.png"
                cv2.imwrite(str(path), frame)
                paths.append(str(path))
            self.keyframes[keyframe] = paths

    def close(self):
        if self.oblique is not None:
            self.oblique.release()
        if self.front is not None:
            self.front.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("pouring", "drawer_open"), required=True)
    parser.add_argument("--snapshot-report", required=True)
    parser.add_argument("--motion-prediction", required=True)
    parser.add_argument("--grasp-object", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--robot-uid",
        choices=("panda", "panda_long_finger", "panda_drawer_finger"),
        default="panda_long_finger",
    )
    parser.add_argument("--pregrasp-distance", type=float, default=0.08)
    parser.add_argument(
        "--approach-gripper-action",
        type=float,
        default=1.0,
        help=(
            "Normalized gripper pre-shape used before and during approach; "
            "-1 is closed and +1 is fully open."
        ),
    )
    parser.add_argument("--grasp-offset-local", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--move-waypoints", type=int, default=24)
    parser.add_argument("--approach-waypoints", type=int, default=14)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--trajectory-substeps", type=int, default=3)
    parser.add_argument("--close-steps", type=int, default=35)
    parser.add_argument("--post-close-hold-steps", type=int, default=10)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--project-drawer-axis", action="store_true")
    parser.add_argument(
        "--drawer-max-pull",
        type=float,
        default=None,
        help="Optional execution-only clamp along the drawer prismatic axis.",
    )
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_report = json.loads(Path(args.snapshot_report).read_text(encoding="utf-8"))
    motion = np.load(Path(args.motion_prediction).expanduser(), allow_pickle=False)
    pose_key = "pred_manipulated_poses_world" if "pred_manipulated_poses_world" in motion else "pred_object_poses_world"
    predicted_raw = np.asarray(motion[pose_key], dtype=np.float32)
    grasp_object = np.load(Path(args.grasp_object).expanduser()).astype(np.float32)

    spec = get_task_spec(args.task)
    extra = {"max_episode_steps": 1200}
    if args.task == "pouring" and "cup_asset" in snapshot_report:
        extra["cup_asset"] = snapshot_report["cup_asset"]
    env = make_env(spec, robot_uids=args.robot_uid, control_mode="pd_ee_pose",
                   render_mode="rgb_array", extra_env_kwargs=extra)
    recorder = Recorder(env, output, args.task, spec.camera_uid, args.fps)
    errors, phases = [], []
    lost_grasp_frame = None
    had_grasp = False

    try:
        obs, _ = env.reset(seed=args.seed, options={"layout": snapshot_report.get("layout", {})})
        unwrapped = env.unwrapped
        manipulated_entity = getattr(unwrapped, spec.manipulated_entity_attr)
        reference_entity = getattr(unwrapped, spec.reference_entity_attr)
        manipulated_initial = _pose_matrix(manipulated_entity.pose)
        reference_initial = _pose_matrix(reference_entity.pose)
        tcp_initial = _pose_matrix(unwrapped.agent.tcp.pose)
        root_world = _pose_matrix(unwrapped.agent.robot.pose)
        alignment_error = float(np.linalg.norm(manipulated_initial[:3, 3] - predicted_raw[0, :3, 3]))
        if alignment_error > 0.005:
            raise RuntimeError(f"Snapshot/execution alignment error {alignment_error:.4f}m")

        predicted = predicted_raw
        projected_scalars = None
        if args.task == "drawer_open" and args.project_drawer_axis:
            predicted, projected_scalars = project_prismatic_trajectory(
                predicted_raw,
                manipulated_initial,
                snapshot_report["pull_axis_world"],
                max_displacement=args.drawer_max_pull,
            )
        tcp_grasp = graspnet_object_row_to_panda_tcp_world(grasp_object, manipulated_initial)
        offset = np.asarray(args.grasp_offset_local, dtype=np.float32)
        tcp_grasp[:3, 3] += tcp_grasp[:3, :3] @ offset
        tcp_pregrasp = pregrasp_pose(tcp_grasp, args.pregrasp_distance)
        tcp_trajectory = object_poses_to_tcp_poses(predicted, manipulated_initial, tcp_grasp)
        last_grasped = False

        def step_to(target, gripper, phase, repeats=1):
            nonlocal obs, last_grasped, lost_grasp_frame, had_grasp
            action = tcp_world_to_absolute_action(target, root_world, gripper)
            for _ in range(repeats):
                obs, _, _, _, info = env.step(action)
                last_grasped = _bool(info["is_grasped"])
                errors.append(float(np.linalg.norm(_pose_matrix(unwrapped.agent.tcp.pose)[:3, 3] - target[:3, 3])))
                if phase == "learned Full64 motion":
                    had_grasp |= last_grasped
                    if had_grasp and not last_grasped and lost_grasp_frame is None:
                        lost_grasp_frame = recorder.frames
                recorder.capture(phase, last_grasped, obs)

        hold_action = tcp_world_to_absolute_action(tcp_initial, root_world, 1.0)
        for i in range(args.hold_steps):
            obs, _, _, _, info = env.step(hold_action)
            last_grasped = _bool(info["is_grasped"])
            recorder.capture("initial", last_grasped, obs, "initial" if i == 0 else None)
        for pose in interpolate_se3(tcp_initial, tcp_pregrasp, args.move_waypoints):
            step_to(
                pose,
                args.approach_gripper_action,
                "move to pregrasp + gripper preshape",
                args.settle_steps,
            )
        recorder.capture("pregrasp reached", last_grasped, keyframe="pregrasp")
        for pose in interpolate_se3(tcp_pregrasp, tcp_grasp, args.approach_waypoints):
            step_to(
                pose,
                args.approach_gripper_action,
                "collision-checked preshaped approach",
                args.settle_steps,
            )
        recorder.capture("grasp pose reached", last_grasped, keyframe="grasp_open")

        qpos_before = _numpy(unwrapped.agent.robot.get_qpos())[-2:].astype(float)
        for _ in range(args.close_steps):
            step_to(tcp_grasp, -1.0, "FULL CLOSE command (-1.0)")
        for _ in range(args.post_close_hold_steps):
            step_to(tcp_grasp, -1.0, "full-close hold")
        qpos_after = _numpy(unwrapped.agent.robot.get_qpos())[-2:].astype(float)
        actual_tcp_after_close = _pose_matrix(unwrapped.agent.tcp.pose)
        close_tcp_error_world = actual_tcp_after_close[:3, 3] - tcp_grasp[:3, 3]
        close_tcp_error_local = tcp_grasp[:3, :3].T @ close_tcp_error_world
        grasp_acquired = last_grasped
        recorder.capture("full close complete", last_grasped, keyframe="grasp_closed")

        previous = tcp_grasp
        for index, target in enumerate(tcp_trajectory):
            for pose in interpolate_se3(previous, target, args.trajectory_substeps):
                step_to(pose, -1.0, "learned Full64 motion")
            previous = target
            if index in {15, 31, 47, 63}:
                recorder.capture(f"Full64 {index + 1}/64", last_grasped,
                                 keyframe=f"trajectory_{index + 1:02d}")
        final_info = None
        for i in range(args.hold_steps):
            obs, _, _, _, final_info = env.step(tcp_world_to_absolute_action(previous, root_world, -1.0))
            last_grasped = _bool(final_info["is_grasped"])
            recorder.capture("final hold", last_grasped, obs,
                             "final" if i == args.hold_steps - 1 else None)

        final_eval = unwrapped.evaluate()
        actual_final = _pose_matrix(manipulated_entity.pose)
        report = {
            "task": args.task,
            "status": "completed_rollout",
            "robot_uid": args.robot_uid,
            "gripper_extension": (
                DEFAULT_LONG_FINGER_SPEC.to_dict()
                if args.robot_uid == "panda_long_finger"
                else DRAWER_LONG_FINGER_SPEC.to_dict()
                if args.robot_uid == "panda_drawer_finger"
                else None
            ),
            "snapshot_execution_alignment_error_m": alignment_error,
            "reference_initial_position_world_m": reference_initial[:3, 3].tolist(),
            "grasp_opening_m": float(grasp_object[1]),
            "approach_gripper_action": args.approach_gripper_action,
            "grasp_acquired_after_close": grasp_acquired,
            "grasped_at_end": last_grasped,
            "lost_grasp_video_frame": lost_grasp_frame,
            "gripper_qpos_before_close_m": qpos_before.tolist(),
            "gripper_qpos_after_close_m": qpos_after.tolist(),
            "target_tcp_grasp_world": tcp_grasp.tolist(),
            "actual_tcp_after_close_world": actual_tcp_after_close.tolist(),
            "close_tcp_error_world_m": close_tcp_error_world.tolist(),
            "close_tcp_error_local_m": close_tcp_error_local.tolist(),
            "drawer_axis_projection_enabled": bool(args.task == "drawer_open" and args.project_drawer_axis),
            "drawer_max_pull_m": args.drawer_max_pull,
            "drawer_projected_final_displacement_m": None if projected_scalars is None else float(projected_scalars[-1]),
            "simulator_success": _bool(final_eval["success"]),
            "drawer_qpos_m": float(_numpy(final_eval["drawer_qpos"])) if "drawer_qpos" in final_eval else None,
            "predicted_final_position_world_m": predicted[-1, :3, 3].tolist(),
            "actual_final_position_world_m": actual_final[:3, 3].tolist(),
            "final_position_error_m": float(np.linalg.norm(actual_final[:3, 3] - predicted[-1, :3, 3])),
            "mean_tcp_tracking_error_m": float(np.mean(errors)),
            "max_tcp_tracking_error_m": float(np.max(errors)),
            "video_frames": recorder.frames,
            "video_fps": args.fps,
            "outputs": {
                "video": str(output / f"{args.task}_execution.mp4"),
                "front_video": str(output / f"{args.task}_execution_front.mp4"),
                "keyframes": recorder.keyframes,
            },
            "phases": phases,
        }
        (output / "execution_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(output / "executed_trajectory.npz",
                            predicted_raw=predicted_raw, predicted_executed=predicted)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        recorder.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
