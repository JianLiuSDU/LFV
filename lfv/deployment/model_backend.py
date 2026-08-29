from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotionPrediction:
    goal_camera: np.ndarray
    object_trajectory_camera: np.ndarray
    metadata: dict[str, Any]


class ExternalMotionBackend:
    """Adapter for an arbitrary model runner writing ``motion_prediction.npz``."""

    def __init__(self, command: str, output_name: str = "motion_prediction.npz"):
        self.command, self.output_name = command, output_name

    def predict(self, *, workdir: str | Path, **_: Any) -> MotionPrediction:
        workdir = Path(workdir)
        subprocess.run(self.command.format(workdir=shlex.quote(str(workdir))), shell=True, check=True, cwd=str(workdir))
        payload = np.load(workdir / self.output_name, allow_pickle=False)
        goal = np.asarray(payload["goal_camera"] if "goal_camera" in payload else payload["goal_pose_camera"], dtype=np.float32).reshape(4, 4)
        trajectory = np.asarray(payload["object_trajectory_camera"], dtype=np.float32).reshape(-1, 4, 4)
        return MotionPrediction(goal, trajectory, {"backend": "external", "artifact": str(workdir / self.output_name)})


class LegacyPouringBackend:
    """Run the existing two-checkpoint pouring adapter in camera coordinates.

    The historical script emits world-frame poses.  The deployment wrapper
    intentionally sets world==camera in its temporary snapshot, so the output
    contract remains camera-frame and the second computer only needs its
    calibrated hand-eye transform.
    """

    def __init__(self, *, model_repo: str, goal_checkpoint: str, trajectory_checkpoint: str, language_embedding: str | None = None, python_executable: str = "python", steps_script: str | None = None, seed: int = 42, device: str = "cuda:0"):
        self.model_repo = Path(model_repo).expanduser()
        self.goal_checkpoint = str(Path(goal_checkpoint).expanduser())
        self.trajectory_checkpoint = str(Path(trajectory_checkpoint).expanduser())
        self.language_embedding = language_embedding
        self.python_executable = python_executable
        self.steps_script = Path(steps_script).expanduser() if steps_script else Path(__file__).resolve().parents[2] / "scripts" / "inference" / "infer_pouring_motion.py"
        self.seed, self.device = int(seed), device

    def predict(self, *, workdir: str | Path, rgb: np.ndarray, depth_m: np.ndarray, cup_mask: np.ndarray, bowl_mask: np.ndarray, intrinsic_cv: np.ndarray, heatmap: np.ndarray) -> MotionPrediction:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        snapshot = workdir / "model_snapshot.npz"
        transfer = workdir / "model_transfer.npz"
        np.savez_compressed(snapshot, rgb=np.asarray(rgb, dtype=np.uint8), depth_m=np.asarray(depth_m, dtype=np.float32), cup_mask=np.asarray(cup_mask, dtype=bool), bowl_mask=np.asarray(bowl_mask, dtype=bool), intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32), T_world_to_camera=np.eye(4, dtype=np.float32), T_object_to_world=np.eye(4, dtype=np.float32))
        np.savez_compressed(transfer, target_heatmap=np.asarray(heatmap, dtype=np.float32))
        output_dir = workdir / "legacy_motion"
        command = [self.python_executable, str(self.steps_script), "--snapshot", str(snapshot), "--transfer-result", str(transfer), "--output-dir", str(output_dir), "--model-repo", str(self.model_repo), "--goal-checkpoint", self.goal_checkpoint, "--trajectory-checkpoint", self.trajectory_checkpoint, "--seed", str(self.seed), "--device", self.device]
        if self.language_embedding:
            command += ["--language-embedding", str(Path(self.language_embedding).expanduser())]
        subprocess.run(command, check=True, cwd=str(self.model_repo))
        payload = np.load(output_dir / "pouring_motion_prediction.npz", allow_pickle=False)
        from lfv.inference.functional_motion.two_stage_pouring import local_pose7d_to_camera_delta
        goal = local_pose7d_to_camera_delta(payload["goal_pose7d_local_stage2"], payload["manipulated_centroid_camera_stage2"])
        trajectory = np.asarray(payload["pred_object_poses_world"], dtype=np.float32)
        return MotionPrediction(goal.astype(np.float32), trajectory, {"backend": "legacy_pouring", "legacy_output": str(output_dir)})


class FunctionalMotionBackend:
    """Run the task-neutral LFV functional-field Goal/Full64 inference."""

    def __init__(self, *, model_repo: str, goal_checkpoint: str, trajectory_checkpoint: str, language_embedding: str, python_executable: str = "python", task: str = "pouring", seed: int = 42, device: str = "cuda:0"):
        self.model_repo = Path(model_repo).expanduser()
        self.goal_checkpoint = str(Path(goal_checkpoint).expanduser())
        self.trajectory_checkpoint = str(Path(trajectory_checkpoint).expanduser())
        self.language_embedding = str(Path(language_embedding).expanduser())
        self.python_executable, self.task, self.seed, self.device = python_executable, task, int(seed), device
        self.steps_script = Path(__file__).resolve().parents[2] / "scripts" / "inference" / "infer_functional_motion.py"

    def predict(self, *, workdir: str | Path, rgb: np.ndarray, depth_m: np.ndarray, cup_mask: np.ndarray, bowl_mask: np.ndarray, intrinsic_cv: np.ndarray, heatmap: np.ndarray) -> MotionPrediction:
        workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        snapshot = workdir / "model_snapshot.npz"; transfer = workdir / "model_transfer.npz"
        np.savez_compressed(snapshot, rgb=np.asarray(rgb, dtype=np.uint8), depth_m=np.asarray(depth_m, dtype=np.float32), cup_mask=np.asarray(cup_mask, dtype=bool), bowl_mask=np.asarray(bowl_mask, dtype=bool), intrinsic_cv=np.asarray(intrinsic_cv, dtype=np.float32), T_world_to_camera=np.eye(4, dtype=np.float32), T_object_to_world=np.eye(4, dtype=np.float32))
        np.savez_compressed(transfer, target_heatmap=np.asarray(heatmap, dtype=np.float32))
        output_dir = workdir / "functional_motion"
        command = [self.python_executable, str(self.steps_script), "--task", self.task, "--snapshot", str(snapshot), "--transfer-result", str(transfer), "--output-dir", str(output_dir), "--model-repo", str(self.model_repo), "--goal-checkpoint", self.goal_checkpoint, "--trajectory-checkpoint", self.trajectory_checkpoint, "--language-embedding", self.language_embedding, "--seed", str(self.seed), "--device", self.device]
        subprocess.run(command, check=True, cwd=str(self.model_repo))
        payload = np.load(output_dir / "functional_motion_prediction.npz", allow_pickle=False)
        from lfv.inference.functional_motion.two_stage import local_pose7d_to_camera_delta
        centroid = np.asarray(payload["manipulated_centroid_camera_stage2"], dtype=np.float32)
        goal = local_pose7d_to_camera_delta(payload["goal_pose7d_local_stage2"], centroid)
        trajectory = np.asarray(payload["pred_object_poses_world"], dtype=np.float32)
        return MotionPrediction(goal.astype(np.float32), trajectory, {"backend": "functional_motion", "functional_output": str(output_dir), "trajectory_steps": int(len(trajectory))})


class FunctionalMotionDirectBackend:
    """Run the current LFV Stage 2 checkpoint directly on camera RGB-D.

    This path preserves the trained model's DINO point features and its
    256-point contract; no cached episode or legacy language-conditioned model
    is involved.
    """

    def __init__(self, *, checkpoint: str, dino_weights: str, device: str = "cpu", seed: int = 42, num_goals: int = 1, num_trajectories: int = 1, motion_memory: str | None = None, motion_field_prior_weight: float = 0.5, fgw_alpha: float = 0.5, fgw_edge_length_ratio: float = 4.0):
        self.checkpoint = Path(checkpoint).expanduser()
        self.dino_weights = Path(dino_weights).expanduser()
        self.device = device
        self.seed, self.num_goals, self.num_trajectories = int(seed), int(num_goals), int(num_trajectories)
        self.motion_memory = Path(motion_memory).expanduser() if motion_memory else None
        self.motion_field_prior_weight = float(motion_field_prior_weight)
        self.fgw_alpha = float(fgw_alpha)
        self.fgw_edge_length_ratio = float(fgw_edge_length_ratio)

    def predict(self, *, workdir: str | Path, rgb: np.ndarray, depth_m: np.ndarray, cup_mask: np.ndarray, bowl_mask: np.ndarray, intrinsic_cv: np.ndarray, heatmap: np.ndarray) -> MotionPrediction:
        import torch
        import torch.nn.functional as F
        from lfv.features.dinov2_dense import DinoV2DenseExtractor
        from lfv.geometry import local_delta_to_camera, pose9d_to_matrix_np
        from lfv.inference.functional_motion.two_stage import localize_clouds, sample_mask_point_cloud
        from lfv.models.functional_motion_generation import load_stage2_checkpoint
        from lfv.models.functional_motion_generation.motion_field_transfer import MotionFieldMemory, transport_motion_field
        from lfv.visualization.motion_field import save_motion_field_comparison

        workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        device = torch.device(self.device)
        model, _, _ = load_stage2_checkpoint(self.checkpoint, device=device, use_ema=True)
        # Stage 1 heat samples are only for grasp instantiation. Stage 2 was
        # trained on the complete manipulated-object mask, so feeding the
        # contact-only subset here would collapse its motion field onto the
        # contact region and corrupt the trajectory distribution.
        m_points, m_pixels = sample_mask_point_cloud(cup_mask, depth_m, intrinsic_cv, 256)
        r_points, r_pixels = sample_mask_point_cloud(bowl_mask, depth_m, intrinsic_cv, 256)
        m_local, r_local, centroid = localize_clouds(m_points, r_points)
        extractor = DinoV2DenseExtractor(model_name="vit_small_patch14_dinov2", weights_path=self.dino_weights, device=self.device)
        patch = int(extractor.patch_size); h, w = rgb.shape[:2]; pad_h = (patch - h % patch) % patch; pad_w = (patch - w % patch) % patch
        padded = np.pad(np.asarray(rgb, dtype=np.uint8), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        grid = extractor.extract(padded)
        def sample_features(pixels: np.ndarray) -> np.ndarray:
            xy = np.asarray(pixels, dtype=np.float32); ph, pw = padded.shape[:2]
            coords = np.stack((2 * xy[:, 0] / max(pw - 1, 1) - 1, 2 * xy[:, 1] / max(ph - 1, 1) - 1), -1)
            feat = torch.from_numpy(grid).permute(2, 0, 1)[None].to(device)
            sampled = F.grid_sample(feat, torch.from_numpy(coords).view(1, len(coords), 1, 2).to(device), mode="bilinear", align_corners=True)
            return F.normalize(sampled.squeeze(0).squeeze(-1).T, dim=-1).cpu().numpy().astype(np.float32)
        manipulated_dino = sample_features(m_pixels)
        reference_dino = sample_features(r_pixels)
        batch = {"manipulated_points": torch.from_numpy(m_local)[None].to(device), "manipulated_dino": torch.from_numpy(manipulated_dino)[None].to(device), "reference_points": torch.from_numpy(r_local)[None].to(device), "reference_dino": torch.from_numpy(reference_dino)[None].to(device)}
        prior_m = prior_r = None
        transfer_payload: dict[str, np.ndarray] = {}
        transfer_confidence = None
        effective_prior_weight = 0.0
        if self.motion_memory is not None:
            memory = MotionFieldMemory.load(self.motion_memory)
            transfer_m = transport_motion_field(memory.manipulated_points, memory.manipulated_dino, memory.manipulated_field, m_local, manipulated_dino, alpha=self.fgw_alpha, edge_length_ratio=self.fgw_edge_length_ratio)
            transfer_r = transport_motion_field(memory.reference_points, memory.reference_dino, memory.reference_field, r_local, reference_dino, alpha=self.fgw_alpha, edge_length_ratio=self.fgw_edge_length_ratio)
            prior_m = torch.from_numpy(transfer_m.target_field)[None].to(device)
            prior_r = torch.from_numpy(transfer_r.target_field)[None].to(device)
            transfer_confidence = float(0.5 * (transfer_m.confidence + transfer_r.confidence))
            effective_prior_weight = float(np.clip(self.motion_field_prior_weight * transfer_confidence, 0.0, 1.0))
            transfer_payload.update({"manipulated_motion_field_prior": transfer_m.target_field, "reference_motion_field_prior": transfer_r.target_field, "manipulated_transport": transfer_m.transport, "reference_transport": transfer_r.transport})
        generator = torch.Generator(device=device).manual_seed(self.seed)
        with torch.inference_mode():
            online_encoding = model.encode(batch, return_debug=True)
            samples, encoding = model.sample(batch, num_goal_samples=self.num_goals, num_trajectory_samples=self.num_trajectories, generator=generator, return_debug=True, motion_field_prior=(prior_m, prior_r) if prior_m is not None else None, motion_field_prior_weight=effective_prior_weight)
        goal_local = samples.goals[0, 0].cpu().numpy().astype(np.float32)
        traj_local = samples.trajectories[0, 0, 0].cpu().numpy().astype(np.float32)
        goal = local_delta_to_camera(pose9d_to_matrix_np(goal_local), centroid, 1.0)
        # Pose9D matrices already contain the local relative transform; convert
        # each around the manipulated centroid with the same Stage 2 adapter.
        trajectory_delta = np.stack(
            [
                local_delta_to_camera(pose9d_to_matrix_np(pose), centroid, 1.0)
                for pose in traj_local
            ],
            axis=0,
        ).astype(np.float32)
        # The diffusion trajectory is an object-relative delta sequence whose
        # first frame is identity. Restore the camera-frame initial object
        # pose before composing the fixed goal-to-gripper attachment; omitting
        # this transform causes a large first-frame jump in the TCP path.
        object_initial_camera = np.eye(4, dtype=np.float32)
        object_initial_camera[:3, 3] = centroid
        trajectory = trajectory_delta @ object_initial_camera
        field_path = workdir / "motion_field.npz"
        payload = {}
        if encoding.manipulated_motion_field is not None: payload["manipulated_motion_field_fused"] = encoding.manipulated_motion_field[0].cpu().numpy()
        if encoding.reference_motion_field is not None: payload["reference_motion_field_fused"] = encoding.reference_motion_field[0].cpu().numpy()
        payload.update(transfer_payload)
        if payload:
            payload.update({"manipulated_points": m_local, "reference_points": r_local, "manipulated_pixels": m_pixels, "reference_pixels": r_pixels})
            np.savez_compressed(field_path, **payload)
        if encoding.manipulated_motion_field is not None and encoding.reference_motion_field is not None:
            manipulated_fused = encoding.manipulated_motion_field[0].cpu().numpy()
            reference_fused = encoding.reference_motion_field[0].cpu().numpy()
            online_m = online_encoding.manipulated_motion_field[0].cpu().numpy() if online_encoding.manipulated_motion_field is not None else manipulated_fused
            online_r = online_encoding.reference_motion_field[0].cpu().numpy() if online_encoding.reference_motion_field is not None else reference_fused
            save_motion_field_comparison(rgb, m_pixels, r_pixels, online_m, online_r, None if prior_m is None else prior_m[0].cpu().numpy(), None if prior_r is None else prior_r[0].cpu().numpy(), manipulated_fused, reference_fused, workdir / "motion_field_comparison.png")
        return MotionPrediction(goal.astype(np.float32), trajectory.astype(np.float32), {"backend": "functional_motion_direct", "checkpoint": str(self.checkpoint), "dino_weights": str(self.dino_weights), "stage2_point_count": 256, "stage2_manipulated_sampling": "full_cup_mask", "stage2_reference_sampling": "full_bowl_mask", "object_trajectory_semantics": "camera_absolute_pose = predicted_delta @ initial_object_pose", "object_initial_camera": object_initial_camera.tolist(), "motion_memory": None if self.motion_memory is None else str(self.motion_memory), "motion_field_prior_weight": self.motion_field_prior_weight, "effective_motion_field_prior_weight": effective_prior_weight, "motion_field_transfer_confidence": transfer_confidence, "motion_field_artifact": str(field_path) if payload else None, "motion_field_comparison": str(workdir / "motion_field_comparison.png") if encoding.manipulated_motion_field is not None else None})
