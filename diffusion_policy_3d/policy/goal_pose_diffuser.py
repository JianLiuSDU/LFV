from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.goal.pose_utils import (
    pose9d_to_matrix,
    pose9d_to_pose7d,
    rotation_geodesic_loss,
    transform_point_cloud,
)
from diffusion_policy_3d.model.goal.relational_pose_encoder import RPDiffStyleRelationEncoder
from diffusion_policy_3d.policy.base_policy import BasePolicy


class GoalPoseDenoiser(nn.Module):
    def __init__(self, encoder_dim=256, goal_dim=9, use_lang_emb=True):
        super().__init__()
        self.encoder = RPDiffStyleRelationEncoder(out_dim=encoder_dim, use_lang_emb=use_lang_emb)
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, goal_dim),
        )

    def forward(
        self,
        noisy_goal_pose9d_norm,
        timestep,
        pc_manipulated,
        pc_target,
        normalizer,
        lang_emb: Optional[torch.Tensor] = None,
    ):
        feat = self.encoder(
            noisy_goal_pose9d_norm=noisy_goal_pose9d_norm,
            timestep=timestep,
            pc_manipulated=pc_manipulated,
            pc_target=pc_target,
            normalizer=normalizer,
            lang_emb=lang_emb,
        )
        return self.head(feat)


class GoalPoseDiffuser(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        num_inference_steps=10,
        goal_dim=9,
        encoder_dim=256,
        use_lang_emb=True,
        loss_weights=None,
        prediction_type="sample",
        **kwargs,
    ):
        super().__init__()
        if prediction_type != "sample":
            raise ValueError("GoalPoseDiffuser first-stage implementation only supports prediction_type='sample'.")
        self.shape_meta = shape_meta
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.goal_dim = goal_dim
        self.encoder_dim = encoder_dim
        self.use_lang_emb = use_lang_emb
        self.prediction_type = prediction_type
        self.loss_weights = {
            "pose9d": 1.0,
            "trans": 1.0,
            "rot": 0.5,
            "cloud": 0.1,
        }
        if loss_weights is not None:
            self.loss_weights.update(dict(loss_weights))
        self.normalizer = LinearNormalizer()
        self.denoiser = GoalPoseDenoiser(encoder_dim=encoder_dim, goal_dim=goal_dim, use_lang_emb=use_lang_emb)

    @staticmethod
    def _squeeze_pc(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"point cloud expected [B,N,3] or [B,1,N,3], got {tuple(x.shape)}")
        return x

    def _normalize_goal(self, x):
        return self.normalizer["goal_pose9d"].normalize(x)

    def _unnormalize_goal(self, x):
        return self.normalizer["goal_pose9d"].unnormalize(x)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        obs = batch["obs"]
        pc_manipulated = self._squeeze_pc(obs["pc_manipulated"])
        pc_target = self._squeeze_pc(obs["pc_target"])
        goal_pose9d = batch["goal_pose9d"]
        lang_emb = obs.get("lang_token_embs", None) if self.use_lang_emb else None

        B = goal_pose9d.shape[0]
        x0_norm = self._normalize_goal(goal_pose9d)
        noise = torch.randn_like(x0_norm)
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, num_train_timesteps, (B,), device=x0_norm.device).long()
        x_t = self.noise_scheduler.add_noise(x0_norm, noise, timesteps)

        pred_x0_norm = self.denoiser(
            noisy_goal_pose9d_norm=x_t,
            timestep=timesteps,
            pc_manipulated=pc_manipulated,
            pc_target=pc_target,
            normalizer=self.normalizer,
            lang_emb=lang_emb,
        )

        loss_pose9d = F.mse_loss(pred_x0_norm, x0_norm)
        pred_pose9d = self._unnormalize_goal(pred_x0_norm)
        gt_pose9d = self._unnormalize_goal(x0_norm)
        T_pred = pose9d_to_matrix(pred_pose9d)
        T_gt = pose9d_to_matrix(gt_pose9d)

        loss_trans = F.smooth_l1_loss(pred_pose9d[:, :3], gt_pose9d[:, :3])
        loss_rot = rotation_geodesic_loss(T_pred[:, :3, :3], T_gt[:, :3, :3])
        P_pred = transform_point_cloud(pc_manipulated, T_pred)
        P_gt = transform_point_cloud(pc_manipulated, T_gt)
        loss_cloud = F.mse_loss(P_pred, P_gt)

        rot_err_rad = self._rotation_error_per_sample(T_pred[:, :3, :3], T_gt[:, :3, :3])
        pos_err_cm = torch.linalg.norm(pred_pose9d[:, :3] - gt_pose9d[:, :3], dim=-1) * 100.0
        loss = (
            self.loss_weights["pose9d"] * loss_pose9d
            + self.loss_weights["trans"] * loss_trans
            + self.loss_weights["rot"] * loss_rot
            + self.loss_weights["cloud"] * loss_cloud
        )
        loss_dict = {
            "loss_pose9d": loss_pose9d.item(),
            "loss_trans": loss_trans.item(),
            "loss_rot_rad": loss_rot.item(),
            "loss_rot_deg": (loss_rot * 180.0 / torch.pi).item(),
            "loss_cloud": loss_cloud.item(),
            "goal_pos_err_cm": pos_err_cm.mean().item(),
            "goal_rot_err_deg": (rot_err_rad.mean() * 180.0 / torch.pi).item(),
        }
        return loss, loss_dict

    @staticmethod
    def _rotation_error_per_sample(R_pred, R_gt):
        R_rel = R_pred.transpose(-1, -2) @ R_gt
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(cos)

    def sample_goal(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pc_manipulated = self._squeeze_pc(obs_dict["pc_manipulated"])
        pc_target = self._squeeze_pc(obs_dict["pc_target"])
        lang_emb = obs_dict.get("lang_token_embs", None) if self.use_lang_emb else None
        B = pc_manipulated.shape[0]
        device = pc_manipulated.device
        dtype = pc_manipulated.dtype
        x = torch.randn(B, self.goal_dim, device=device, dtype=dtype)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((B,), int(t), device=device, dtype=torch.long)
            pred_x0_norm = self.denoiser(
                noisy_goal_pose9d_norm=x,
                timestep=timesteps,
                pc_manipulated=pc_manipulated,
                pc_target=pc_target,
                normalizer=self.normalizer,
                lang_emb=lang_emb,
            )
            x = self.noise_scheduler.step(pred_x0_norm, t, x).prev_sample

        pose9d = self._unnormalize_goal(x)
        pose7d = pose9d_to_pose7d(pose9d)
        T_goal = pose9d_to_matrix(pose9d)
        return {
            "goal_pose9d": pose9d,
            "goal_pose7d": pose7d,
            "T_goal": T_goal,
        }

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.sample_goal(obs_dict)
