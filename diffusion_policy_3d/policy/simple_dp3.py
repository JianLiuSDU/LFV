from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
#from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d_progress import ConditionalUnet1D_progress
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
#from diffusion_policy_3d.model.vision.cross_attention_encoder import CrossAttentionDP3Encoder

#----------------------------------------------------
# 条件引导部分的引入
from diffusion_policy_3d.model.vision.cross_attention_encoder import (
    ManipulationCentricSE3Encoder,
    GoalConditionedSetTransformerEncoder,
)
#----------------------------------------------------



# def custom_normalize(data, normalizer, key):
#     pose = copy.deepcopy(data) # (B, T, 7) or dict
    
#     # 💡 屏蔽刷屏警告：把不需要归一化的 key (如点云、语言) 暂时抽出来
#     ignored_dict = {}
#     if isinstance(pose, dict):
#         keys_to_ignore = [k for k in pose.keys() if k not in ['agent_pos', 'action']]
#         for k in keys_to_ignore:
#             ignored_dict[k] = pose.pop(k)

#     if key is not None:
#         use_progress = True if pose[key].shape[-1] > 7 else False
#     else:
#         use_progress = True if pose.shape[-1] > 7 else False

#     if key is not None:
#         device = pose[key].device
#         # save and normalize quaternion
#         if use_progress:
#             progress = pose[key][:, :, 7:8]
#         pose_ori = pose[key][:, :, 3:7] 
#         # remove quaternion
#         pose[key] = pose[key][:, :, :3]
#     else:
#         device = pose.device
#         # save and normalize quaternion
#         if use_progress:
#             progress = pose[:, :, 7:8] 
#         pose_ori = pose[:, :, 3:7] 
#         # remove quaternion
#         pose = pose[:, :, :3]

#     # normalize data and quaternion
#     pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
#     npose = normalizer.normalize(pose)

#     if key is not None:
#         pose_ori = pose_ori.to(npose[key].device)
#         # add the normalize quaternion back
#         if use_progress:
#             progress = progress.to(npose[key].device)
#             npose[key] = torch.concat([npose[key], pose_ori, progress], dim=-1)
#         else:
#             npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
#     else:
#         pose_ori = pose_ori.to(npose.device)
#         if use_progress:
#             progress = progress.to(npose.device)
#             npose = torch.concat([npose, pose_ori, progress], dim=-1)
#         else:
#             npose = torch.concat([npose, pose_ori], dim=-1)
            
#     # 💡 屏蔽刷屏警告：归一化结束后，把点云等数据塞回去
#     if isinstance(npose, dict):
#         for k, v in ignored_dict.items():
#             npose[k] = v
            
#     return npose
def custom_normalize(data, normalizer, key):
    pose = copy.deepcopy(data)

    ignored_dict = {}
    if isinstance(pose, dict):
        keys_to_ignore = [k for k in pose.keys() if k not in ['agent_pos', 'action']]
        for k in keys_to_ignore:
            ignored_dict[k] = pose.pop(k)

    if key is not None:
        pose_ori = pose[key][:, :, 3:7]
        pose[key] = pose[key][:, :, :3]
    else:
        pose_ori = pose[:, :, 3:7]
        pose = pose[:, :, :3]

    pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
    npose = normalizer.normalize(pose)

    if key is not None:
        pose_ori = pose_ori.to(npose[key].device)
        npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
    else:
        pose_ori = pose_ori.to(npose.device)
        npose = torch.concat([npose, pose_ori], dim=-1)

    if isinstance(npose, dict):
        for k, v in ignored_dict.items():
            npose[k] = v

    return npose

# def custom_unnormalize(data, normalizer, key):
#     pose = copy.deepcopy(data) # (B, T, 7) or dict
    
#     # 💡 屏蔽刷屏警告：把不需要反归一化的 key 暂时抽出来
#     ignored_dict = {}
#     if isinstance(pose, dict):
#         keys_to_ignore = [k for k in pose.keys() if k not in ['agent_pos', 'action']]
#         for k in keys_to_ignore:
#             ignored_dict[k] = pose.pop(k)

#     if key is not None:
#         use_progress = True if pose[key].shape[-1] > 7 else False
#     else:
#         use_progress = True if pose.shape[-1] > 7 else False

#     if key is not None:
#         # save and normalize quaternion
#         if use_progress:
#             progress = pose[key][:, :, 7:8]
#         pose_ori = pose[key][:, :, 3:7] 
#         # remove quaternion
#         pose[key] = pose[key][:, :, :3]
#     else:
#         # save and normalize quaternion
#         if use_progress:
#             progress = pose[:, :, 7:8] 
#         pose_ori = pose[:, :, 3:7] 
#         # remove quaternion
#         pose = pose[:, :, :3]

#     # normalize data and quaternion
#     pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
#     npose = normalizer.unnormalize(pose)

#     if key is not None:
#         # add the normalize quaternion back
#         if use_progress:
#             npose[key] = torch.concat([npose[key], pose_ori, progress], dim=-1)
#         else:
#             npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
#     else:
#         if use_progress:
#             npose = torch.concat([npose, pose_ori, progress], dim=-1)
#         else:
#             npose = torch.concat([npose, pose_ori], dim=-1)
            
#     # 💡 屏蔽刷屏警告：反归一化结束后，把点云等数据塞回去
#     if isinstance(npose, dict):
#         for k, v in ignored_dict.items():
#             npose[k] = v
            
#     return npose
def custom_unnormalize(data, normalizer, key):
    pose = copy.deepcopy(data)

    ignored_dict = {}
    if isinstance(pose, dict):
        keys_to_ignore = [k for k in pose.keys() if k not in ['agent_pos', 'action']]
        for k in keys_to_ignore:
            ignored_dict[k] = pose.pop(k)

    if key is not None:
        pose_ori = pose[key][:, :, 3:7]
        pose[key] = pose[key][:, :, :3]
    else:
        pose_ori = pose[:, :, 3:7]
        pose = pose[:, :, :3]

    pose_ori = torch.nn.functional.normalize(pose_ori, dim=2)
    npose = normalizer.unnormalize(pose)

    if key is not None:
        npose[key] = torch.concat([npose[key], pose_ori], dim=-1)
    else:
        npose = torch.concat([npose, pose_ori], dim=-1)

    if isinstance(npose, dict):
        for k, v in ignored_dict.items():
            npose[k] = v

    return npose



class SimpleDP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            use_cross_attention=True,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            use_lang_emb=False,
            use_stage_emb=False,
            use_progress=False,
            encoder_output_dim=256,
            obs_encoder_type="manipulation_centric_se3",
            use_goal_condition=False,
            goal_pose_key="goal_pose9d",
            goal_delta_pose_key="goal_delta_pose9d",
            goal_delta_pose7d_key="goal_delta_pose7d",
            goal_noise_std_xyz=0.0,
            goal_noise_std_rot=0.0,
            use_boundary_inpainting=False,
            start_boundary_mode="identity",
            end_boundary_mode="goal_delta_pose7d",
            goal_encoder_cfg=None,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            predict_type = "relative",
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.obs_encoder_type = obs_encoder_type
        self.use_goal_condition = use_goal_condition
        self.goal_pose_key = goal_pose_key
        self.goal_delta_pose_key = goal_delta_pose_key
        self.goal_delta_pose7d_key = goal_delta_pose7d_key
        self.goal_noise_std_xyz = goal_noise_std_xyz
        self.goal_noise_std_rot = goal_noise_std_rot
        self.use_boundary_inpainting = use_boundary_inpainting
        self.start_boundary_mode = start_boundary_mode
        self.end_boundary_mode = end_boundary_mode

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        # self.use_progress = use_progress
        # if use_progress:
        #     assert len(action_shape) == 1 and action_shape[0] == 8
        # else:
        #     assert len(action_shape) == 1 and action_shape[0] == 7
        self.use_progress = False
        assert len(action_shape) == 1 and action_shape[0] == 7
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # if use_cross_attention:
        #     cprint("[SDP3] 🚀 启用双点云 Cross-Attention 编码器", "cyan", attrs=['bold'])
        #     obs_encoder = CrossAttentionDP3Encoder(
        #         observation_space=obs_dict,
        #         out_channel=encoder_output_dim,
        #         use_lang_emb=use_lang_emb
        #     )
        
        
        #----------------------------------------------------
        # 条件引导部分的引入
        if obs_encoder_type == "goal_conditioned_set_transformer":
            cprint("[SDP3] 🚀 启用 Goal-Conditioned Set Transformer 编码器", "cyan", attrs=['bold'])
            goal_encoder_cfg = goal_encoder_cfg or {}
            obs_encoder = GoalConditionedSetTransformerEncoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                use_lang_emb=use_lang_emb,
                **goal_encoder_cfg,
            )
        elif use_cross_attention:
            cprint("[SDP3] 🚀 启用 Manipulation-Centric SE3 编码器", "cyan", attrs=['bold'])
            obs_encoder = ManipulationCentricSE3Encoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                use_lang_emb=use_lang_emb,
                d_pc=128,
                d_lang=128,
                d_pos=64,
                d_geom=64,
                pe_bands=6,
                k_rel=8
            )        
        #----------------------------------------------------
            
        else:
            cprint("[SDP3] 🔹 使用单点云基础编码器 (DP3Encoder)", "white")
            obs_encoder = DP3Encoder(
                observation_space=obs_dict,
                img_crop_shape=crop_shape,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                use_lang_emb=use_lang_emb,
                use_stage_emb=use_stage_emb,
            )

        cprint(f"[SDP3] use_lang_emb: {use_lang_emb}", "yellow")
        cprint(f"[SDP3] use_stage_emb: {use_stage_emb}", "yellow")

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            # 无论什么 condition_type，这里直接给定最终展平的维度
            global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SDP3] pointnet_type: {self.pointnet_type}", "yellow")

        # if use_progress:
        #     input_dim = input_dim - 1 # handle progress/gripper separately
        #     model = ConditionalUnet1D_progress(
        #         input_dim=input_dim,
        #         local_cond_dim=None,
        #         global_cond_dim=global_cond_dim,
        #         diffusion_step_embed_dim=diffusion_step_embed_dim,
        #         down_dims=down_dims,
        #         kernel_size=kernel_size,
        #         n_groups=n_groups,
        #         condition_type=condition_type,
        #         use_down_condition=use_down_condition,
        #         use_mid_condition=use_mid_condition,
        #         use_up_condition=use_up_condition,
        #     )
        # else:
        #     model = ConditionalUnet1D(
        #         input_dim=input_dim,
        #         local_cond_dim=None,
        #         global_cond_dim=global_cond_dim,
        #         diffusion_step_embed_dim=diffusion_step_embed_dim,
        #         down_dims=down_dims,
        #         kernel_size=kernel_size,
        #         n_groups=n_groups,
        #         condition_type=condition_type,
        #         use_down_condition=use_down_condition,
        #         use_mid_condition=use_mid_condition,
        #         use_up_condition=use_up_condition,
        #     )
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.predict_type = predict_type

        print_params(self)

    def _normalize_goal_obs(self, obs_dict, add_noise=False):
        if not self.use_goal_condition:
            return obs_dict

        nobs = dict(obs_dict)
        if self.goal_pose_key in nobs:
            goal_pose = nobs[self.goal_pose_key]
            if add_noise and self.goal_noise_std_xyz > 0:
                goal_pose = goal_pose.clone()
                goal_pose[..., :3] = goal_pose[..., :3] + torch.randn_like(goal_pose[..., :3]) * self.goal_noise_std_xyz
            nobs[self.goal_pose_key] = self.normalizer[self.goal_pose_key].normalize(goal_pose)

        if self.goal_delta_pose_key in nobs:
            goal_delta = nobs[self.goal_delta_pose_key]
            if add_noise and self.goal_noise_std_xyz > 0:
                goal_delta = goal_delta.clone()
                goal_delta[..., :3] = goal_delta[..., :3] + torch.randn_like(goal_delta[..., :3]) * self.goal_noise_std_xyz
            nobs[self.goal_delta_pose_key] = self.normalizer[self.goal_delta_pose_key].normalize(goal_delta)

        return nobs

    def build_boundary_condition(self, obs_dict, shape, device, dtype):
        condition_data = torch.zeros(size=shape, device=device, dtype=dtype)
        condition_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)

        if not self.use_boundary_inpainting:
            return condition_data, condition_mask

        if self.start_boundary_mode != "identity":
            raise ValueError(f"Unsupported start_boundary_mode: {self.start_boundary_mode}")
        if self.end_boundary_mode != "goal_delta_pose7d":
            raise ValueError(f"Unsupported end_boundary_mode: {self.end_boundary_mode}")
        if self.goal_delta_pose7d_key not in obs_dict:
            raise KeyError(f"Missing obs['{self.goal_delta_pose7d_key}'] for boundary inpainting")

        B, T, Da = shape
        identity = torch.zeros((B, 1, Da), device=device, dtype=dtype)
        identity[..., 6] = 1.0

        goal_delta = obs_dict[self.goal_delta_pose7d_key].to(device=device, dtype=dtype)
        if goal_delta.dim() == 2:
            goal_delta = goal_delta.unsqueeze(1)
        elif goal_delta.dim() > 3:
            goal_delta = goal_delta.reshape(B, -1, Da)[:, -1:, :]
        else:
            goal_delta = goal_delta[:, -1:, :]

        boundary_poses = torch.cat([identity, goal_delta], dim=1)
        n_boundary = custom_normalize(boundary_poses, self.normalizer["action"], key=None)

        condition_data[:, 0, :] = n_boundary[:, 0, :]
        condition_data[:, -1, :] = n_boundary[:, 1, :]
        condition_mask[:, 0, :] = True
        condition_mask[:, -1, :] = True
        return condition_data, condition_mask
        
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            trajectory = scheduler.step(
                model_output, t, trajectory).prev_sample
            trajectory[condition_mask] = condition_data[condition_mask]
                
        trajectory[condition_mask] = condition_data[condition_mask]   
        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], target=None) -> Dict[str, torch.Tensor]:
        nobs = custom_normalize(obs_dict, self.normalizer, key='agent_pos')
        nobs = self._normalize_goal_obs(nobs, add_noise=False)

        if 'point_cloud' in nobs:
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        value = nobs['agent_pos'] if 'agent_pos' in nobs else next(iter(nobs.values()))

        if value.shape[1] > 1 :
            value = value[:, 0].unsqueeze(1)

        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            
            # 💡 强行展平为 2D 张量，避免维度报错
            global_cond = nobs_features.reshape(B, -1)
            
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            if self.use_boundary_inpainting:
                cond_data, cond_mask = self.build_boundary_condition(nobs, (B, T, Da), device, dtype)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        naction_pred = nsample[...,:Da]
        action_pred = custom_unnormalize(naction_pred, self.normalizer['action'], key=None)
        
        start = 0 if self.use_boundary_inpainting else To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }

        if target is not None:
            action_pred = action_pred.to(target.device)
            mse = torch.nn.functional.mse_loss(action_pred, target)
            result["loss"] = mse.item()
        return result


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(self, batch):
        nobs = custom_normalize(batch['obs'], self.normalizer, key='agent_pos')
        nobs = self._normalize_goal_obs(nobs, add_noise=True)
        nactions = custom_normalize(batch['action'], self.normalizer['action'], key=None)

        if 'point_cloud' in nobs:
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            # 💡 强行展平为 2D 张量，完美对接 FiLM 机制
            global_cond = nobs_features.reshape(batch_size, -1)

            if 'point_cloud' in nobs:
                this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
                this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        if self.use_boundary_inpainting:
            cond_data, condition_mask = self.build_boundary_condition(
                nobs, trajectory.shape, trajectory.device, trajectory.dtype
            )
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                        local_cond=local_cond, 
                        global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        # # 💡 【史诗级修复】：扩散模型预测噪声永远只用 MSE
        # if self.use_progress:
        #     loss_trans = F.mse_loss(pred[..., :3], target[..., :3], reduction='none')
        #     loss_ori = F.mse_loss(pred[..., 3:-1], target[..., 3:-1], reduction='none')
        #     loss_gripper = F.mse_loss(pred[..., -1:], target[..., -1:], reduction='none')

        #     # 拼接 8D 损失
        #     loss = torch.concat([loss_trans, loss_ori, loss_gripper], dim=-1)
            
        #     # 降低夹爪维度的 Loss 权重，防止它干扰主要的空间运动学习
        #     loss[..., -1:] *= 0.1
        # else:
        #     loss = F.mse_loss(pred, target, reduction='none')
        loss = F.mse_loss(pred, target, reduction='none')

        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.sum() / loss_mask.type(loss.dtype).sum().clamp_min(1.0)

        loss_dict = {
            'bc_loss': loss.item(),
        }
        
        return loss, loss_dict

        # if self.use_progress:
        #     loss_trans = F.mse_loss(pred[..., :3], target[..., :3], reduction='none')
        #     loss_ori = F.mse_loss(pred[..., 3:-1], target[..., 3:-1], reduction='none')
        #     loss_gripper = F.binary_cross_entropy(pred[..., -1:], target[..., -1:], reduction='none')

        #     loss = torch.concat([loss_trans, loss_ori, loss_gripper], dim=-1)
        #     loss[..., -1:] *= 0.1
        # else:
        #     loss = F.mse_loss(pred, target, reduction='none')

        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()

        # loss_dict = {
        #     'bc_loss': loss.item(),
        # }
        
        # return loss, loss_dict
