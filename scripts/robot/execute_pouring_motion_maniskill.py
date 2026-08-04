#!/usr/bin/env python3
"""Execute a saved GraspNet grasp and learned cup trajectory in ManiSkill."""

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

from lfv.robot.panda_grasp_execution import (
    graspnet_object_row_to_panda_tcp_world,
    interpolate_se3,
    maniskill_wxyz_pose_to_matrix,
    object_poses_to_tcp_poses,
    pregrasp_pose,
    tcp_world_to_absolute_action,
)
from lfv.robot.gripper_extension import DEFAULT_LONG_FINGER_SPEC
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


def _frame_rgb(env) -> np.ndarray:
    frame = _numpy(env.render())
    if frame.ndim == 4:
        frame = frame[0]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255)
    return np.asarray(frame, dtype=np.uint8)


def _front_frame_rgb(obs: dict, camera_uid: str) -> np.ndarray:
    return extract_camera_observation(obs, camera_uid).rgb


def _pose_matrix(pose_struct) -> np.ndarray:
    return maniskill_wxyz_pose_to_matrix(_numpy(pose_struct.raw_pose))


class Recorder:
    def __init__(self, env, output_dir: Path, fps: int, front_camera_uid: str):
        self.env = env
        self.output_dir = output_dir
        self.fps = fps
        self.front_camera_uid = front_camera_uid
        self.oblique_writer = None
        self.front_writer = None
        self.frame_count = 0
        self.keyframes_oblique: dict[str, str] = {}
        self.keyframes_front: dict[str, str] = {}
        self.last_front_rgb = None

    @staticmethod
    def _annotate(rgb: np.ndarray, phase: str, grasped: bool, view: str) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 72), (18, 18, 18), -1)
        cv2.putText(
            bgr,
            f"{view} | phase: {phase}",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            f"pouring model + GraspNet | grasped: {str(grasped).lower()}",
            (18, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (70, 220, 255) if grasped else (160, 160, 255),
            2,
            cv2.LINE_AA,
        )
        return bgr

    @staticmethod
    def _open_writer(path: Path, fps: int, bgr: np.ndarray):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (bgr.shape[1], bgr.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        return writer

    def capture(
        self,
        phase: str,
        grasped: bool,
        *,
        obs: dict | None = None,
        keyframe: str | None = None,
    ):
        oblique_bgr = self._annotate(
            _frame_rgb(self.env), phase, grasped, "oblique view"
        )
        if obs is not None:
            self.last_front_rgb = _front_frame_rgb(obs, self.front_camera_uid)
        if self.last_front_rgb is None:
            raise RuntimeError("A front-camera observation is required before recording")
        front_bgr = self._annotate(
            self.last_front_rgb, phase, grasped, "front view"
        )
        if self.oblique_writer is None:
            self.oblique_writer = self._open_writer(
                self.output_dir / "pouring_model_execution.mp4",
                self.fps,
                oblique_bgr,
            )
            self.front_writer = self._open_writer(
                self.output_dir / "pouring_model_execution_front.mp4",
                self.fps,
                front_bgr,
            )
        self.oblique_writer.write(oblique_bgr)
        assert self.front_writer is not None
        self.front_writer.write(front_bgr)
        self.frame_count += 1
        if keyframe is not None:
            oblique_path = self.output_dir / f"keyframe_{keyframe}.png"
            front_path = self.output_dir / f"keyframe_front_{keyframe}.png"
            cv2.imwrite(str(oblique_path), oblique_bgr)
            cv2.imwrite(str(front_path), front_bgr)
            self.keyframes_oblique[keyframe] = str(oblique_path)
            self.keyframes_front[keyframe] = str(front_path)

    def close(self):
        if self.oblique_writer is not None:
            self.oblique_writer.release()
        if self.front_writer is not None:
            self.front_writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-report", required=True)
    parser.add_argument("--motion-prediction", required=True)
    parser.add_argument("--grasp-object", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--robot-uid",
        choices=("panda", "panda_long_finger"),
        default="panda",
    )
    parser.add_argument("--pregrasp-distance", type=float, default=0.10)
    parser.add_argument(
        "--grasp-offset-local",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ORTHOGONAL", "CLOSING", "APPROACH"),
        help="TCP translation correction in Panda grasp-frame meters.",
    )
    parser.add_argument("--move-waypoints", type=int, default=35)
    parser.add_argument("--approach-waypoints", type=int, default=18)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--trajectory-substeps", type=int, default=3)
    parser.add_argument("--close-steps", type=int, default=24)
    parser.add_argument("--post-close-hold-steps", type=int, default=15)
    parser.add_argument("--hold-steps", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_report = json.loads(
        Path(args.snapshot_report).expanduser().read_text(encoding="utf-8")
    )
    layout = snapshot_report["layout"]
    cup_asset = snapshot_report["cup_asset"]
    motion = np.load(Path(args.motion_prediction).expanduser(), allow_pickle=False)
    predicted_object_poses = np.asarray(
        motion["pred_object_poses_world"], dtype=np.float32
    )
    predicted_object_initial = predicted_object_poses[0]
    grasp_row_object = np.load(Path(args.grasp_object).expanduser()).astype(np.float32)

    spec = get_task_spec("pouring")
    env = make_env(
        spec,
        robot_uids=args.robot_uid,
        control_mode="pd_ee_pose",
        render_mode="rgb_array",
        extra_env_kwargs={
            "cup_asset": cup_asset,
            "max_episode_steps": 1000,
        },
    )
    recorder = Recorder(env, output_dir, args.fps, spec.camera_uid)
    phase_log: list[dict] = []
    tracking_errors = []
    lost_grasp_step = None
    had_trajectory_grasp = False

    try:
        reset_obs, _ = env.reset(seed=args.seed, options={"layout": layout})
        unwrapped = env.unwrapped
        object_initial = _pose_matrix(unwrapped.cup.pose)
        initial_bowl = _pose_matrix(unwrapped.bowl.pose)
        tcp_initial = _pose_matrix(unwrapped.agent.tcp.pose)
        root_world = _pose_matrix(unwrapped.agent.robot.pose)
        snapshot_alignment_error = float(
            np.linalg.norm(object_initial[:3, 3] - predicted_object_initial[:3, 3])
        )
        if snapshot_alignment_error > 0.005:
            raise RuntimeError(
                "Execution scene does not match inference snapshot: "
                f"initial cup error={snapshot_alignment_error:.4f} m"
            )

        tcp_grasp = graspnet_object_row_to_panda_tcp_world(
            grasp_row_object, object_initial
        )
        grasp_offset_local = np.asarray(args.grasp_offset_local, dtype=np.float32)
        tcp_grasp[:3, 3] += tcp_grasp[:3, :3] @ grasp_offset_local
        tcp_pregrasp = pregrasp_pose(tcp_grasp, args.pregrasp_distance)
        predicted_tcp_poses = object_poses_to_tcp_poses(
            predicted_object_poses, object_initial, tcp_grasp
        )
        # The arm part of pd_ee_pose is absolute/unscaled, while the Panda
        # gripper part is normalized: +1=fully open and -1=full close.
        open_gripper_action = 1.0
        full_close_gripper_action = -1.0

        last_grasped = False

        def step_to(tcp_world: np.ndarray, gripper: float, phase: str, repeats: int = 1):
            nonlocal last_grasped, lost_grasp_step, had_trajectory_grasp
            action = tcp_world_to_absolute_action(tcp_world, root_world, gripper)
            for _ in range(repeats):
                obs, _, _, _, info = env.step(action)
                last_grasped = _bool(info["is_grasped"])
                actual_tcp = _pose_matrix(unwrapped.agent.tcp.pose)
                tracking_errors.append(
                    float(np.linalg.norm(actual_tcp[:3, 3] - tcp_world[:3, 3]))
                )
                if phase == "learned Full64 trajectory":
                    if last_grasped:
                        had_trajectory_grasp = True
                    elif had_trajectory_grasp and lost_grasp_step is None:
                        lost_grasp_step = recorder.frame_count
                recorder.capture(phase, last_grasped, obs=obs)

        current_hold_action = tcp_world_to_absolute_action(
            tcp_initial, root_world, open_gripper_action
        )
        for index in range(args.hold_steps):
            obs, _, _, _, info = env.step(current_hold_action)
            last_grasped = _bool(info["is_grasped"])
            recorder.capture(
                "initial scene / model inputs",
                last_grasped,
                obs=obs,
                keyframe="initial" if index == 0 else None,
            )
        phase_log.append({"phase": "initial_hold", "frames": args.hold_steps})

        for pose in interpolate_se3(tcp_initial, tcp_pregrasp, args.move_waypoints):
            step_to(
                pose,
                open_gripper_action,
                "move to GraspNet pregrasp",
                args.settle_steps,
            )
        recorder.capture("GraspNet pregrasp reached", last_grasped, keyframe="pregrasp")
        phase_log.append({"phase": "move_pregrasp", "waypoints": args.move_waypoints})

        for pose in interpolate_se3(tcp_pregrasp, tcp_grasp, args.approach_waypoints):
            step_to(pose, open_gripper_action, "top-down approach", args.settle_steps)
        recorder.capture("top-down grasp pose reached", last_grasped, keyframe="grasp_open")
        phase_log.append({"phase": "approach", "waypoints": args.approach_waypoints})

        gripper_qpos_before_close = _numpy(unwrapped.agent.robot.get_qpos())[-2:].astype(float)
        for _ in range(args.close_steps):
            step_to(tcp_grasp, full_close_gripper_action, "FULL CLOSE command (-1.0)", 1)
        for _ in range(args.post_close_hold_steps):
            step_to(
                tcp_grasp,
                full_close_gripper_action,
                "hold FULL CLOSE before learned motion",
                1,
            )
        gripper_qpos_after_close = _numpy(unwrapped.agent.robot.get_qpos())[-2:].astype(float)
        actual_tcp_after_close = _pose_matrix(unwrapped.agent.tcp.pose)
        tcp_error_world_after_close = (
            actual_tcp_after_close[:3, 3] - tcp_grasp[:3, 3]
        )
        tcp_error_local_after_close = (
            tcp_grasp[:3, :3].T @ tcp_error_world_after_close
        )
        grasp_acquired = last_grasped
        recorder.capture("full-close hold completed", last_grasped, keyframe="grasp_closed")
        phase_log.append(
            {
                "phase": "explicit_full_close",
                "normalized_command": full_close_gripper_action,
                "close_steps": args.close_steps,
                "post_close_hold_steps": args.post_close_hold_steps,
                "gripper_qpos_before_m": gripper_qpos_before_close.tolist(),
                "gripper_qpos_after_m": gripper_qpos_after_close.tolist(),
                "tcp_error_world_after_close_m": tcp_error_world_after_close.tolist(),
                "tcp_error_local_after_close_m": tcp_error_local_after_close.tolist(),
                "grasp_acquired": grasp_acquired,
            }
        )

        previous = tcp_grasp
        for trajectory_index, target in enumerate(predicted_tcp_poses):
            for pose in interpolate_se3(previous, target, args.trajectory_substeps):
                step_to(pose, full_close_gripper_action, "learned Full64 trajectory", 1)
            previous = target
            if trajectory_index in {15, 31, 47, 63}:
                recorder.capture(
                    f"learned trajectory {trajectory_index + 1}/64",
                    last_grasped,
                    keyframe=f"trajectory_{trajectory_index + 1:02d}",
                )
        phase_log.append(
            {
                "phase": "learned_trajectory",
                "model_waypoints": int(len(predicted_tcp_poses)),
                "substeps": args.trajectory_substeps,
            }
        )

        final_action = tcp_world_to_absolute_action(
            previous, root_world, full_close_gripper_action
        )
        final_info = None
        for index in range(args.hold_steps):
            obs, _, _, _, final_info = env.step(final_action)
            last_grasped = _bool(final_info["is_grasped"])
            recorder.capture(
                "final hold",
                last_grasped,
                obs=obs,
                keyframe="final" if index == args.hold_steps - 1 else None,
            )
        actual_object_final = _pose_matrix(unwrapped.cup.pose)
        actual_tcp_final = _pose_matrix(unwrapped.agent.tcp.pose)
        final_eval = unwrapped.evaluate()
        video_path = output_dir / "pouring_model_execution.mp4"
        front_video_path = output_dir / "pouring_model_execution_front.mp4"
        report = {
            "robot_uid": args.robot_uid,
            "gripper_extension": (
                DEFAULT_LONG_FINGER_SPEC.to_dict()
                if args.robot_uid == "panda_long_finger"
                else None
            ),
            "status": "completed_rollout",
            "note": (
                "Rerun with corrected normalized full-close control, optional "
                "long-finger collision geometry, and synchronized oblique/front video."
            ),
            "model": "trained pouring goal-pose diffuser + trained pouring Full64 diffuser",
            "grasp_source": "saved contact-constrained top-down GraspNet pose",
            "layout": layout,
            "initial_cup_bowl_planar_distance_m": float(
                np.linalg.norm(object_initial[:2, 3] - initial_bowl[:2, 3])
            ),
            "snapshot_execution_alignment_error_m": snapshot_alignment_error,
            "grasp_opening_m": float(grasp_row_object[1]),
            "grasp_offset_local_m": grasp_offset_local.tolist(),
            "gripper_control_contract": {
                "arm": "absolute base-frame XYZ + Euler XYZ",
                "gripper": "normalized scalar: +1 fully open, -1 full close",
                "full_close_command": full_close_gripper_action,
                "actual_qpos_before_close_m": gripper_qpos_before_close.tolist(),
                "actual_qpos_after_close_m": gripper_qpos_after_close.tolist(),
            },
            "grasp_acquired_after_close": grasp_acquired,
            "grasped_at_end": last_grasped,
            "lost_grasp_video_frame": lost_grasp_step,
            "simulator_success": _bool(final_eval["success"]),
            "cup_above_bowl": _bool(final_eval["is_cup_above_bowl"]),
            "predicted_final_object_position_world_m": predicted_object_poses[-1, :3, 3].tolist(),
            "actual_final_object_position_world_m": actual_object_final[:3, 3].tolist(),
            "final_object_position_tracking_error_m": float(
                np.linalg.norm(actual_object_final[:3, 3] - predicted_object_poses[-1, :3, 3])
            ),
            "final_tcp_position_tracking_error_m": float(
                np.linalg.norm(actual_tcp_final[:3, 3] - previous[:3, 3])
            ),
            "mean_tcp_position_tracking_error_m": float(np.mean(tracking_errors)),
            "max_tcp_position_tracking_error_m": float(np.max(tracking_errors)),
            "video_frames": recorder.frame_count,
            "video_fps": args.fps,
            "phases": phase_log,
            "outputs": {
                "video": str(video_path),
                "front_video": str(front_video_path),
                "keyframes_oblique": recorder.keyframes_oblique,
                "keyframes_front": recorder.keyframes_front,
            },
        }
        (output_dir / "execution_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        recorder.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
