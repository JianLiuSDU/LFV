import math
from typing import Optional

import torch
import torch.nn as nn

from diffusion_policy_3d.model.goal.pose_utils import pose9d_to_matrix, transform_point_cloud


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class RelationBranch(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        self.child_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.child_role_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.target_role_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.out_dim = embed_dim * 2 + 3

    @staticmethod
    def _normalize_pair(pc_child: torch.Tensor, pc_target: torch.Tensor):
        c_child = pc_child.mean(dim=1, keepdim=True)
        c_tgt = pc_target.mean(dim=1, keepdim=True)
        child_local = pc_child - c_child
        target_local = pc_target - c_tgt
        rel_center = (c_tgt - c_child).squeeze(1)

        bbox_min = pc_target.min(dim=1).values
        bbox_max = pc_target.max(dim=1).values
        scale = torch.linalg.norm(bbox_max - bbox_min, dim=-1, keepdim=True).clamp_min(1e-4)
        child_local = child_local / scale[:, None, :]
        target_local = target_local / scale[:, None, :]
        rel_center = rel_center / scale
        return child_local, target_local, rel_center

    def forward(self, pc_child: torch.Tensor, pc_target: torch.Tensor) -> torch.Tensor:
        child_local, target_local, rel_center = self._normalize_pair(pc_child, pc_target)
        child_tokens = self.child_mlp(child_local) + self.child_role_emb
        target_tokens = self.target_mlp(target_local) + self.target_role_emb
        relation_tokens, _ = self.cross_attn(child_tokens, target_tokens, target_tokens, need_weights=False)
        relation_tokens = relation_tokens + self.ffn(relation_tokens)
        pooled = torch.cat(
            [relation_tokens.max(dim=1).values, relation_tokens.mean(dim=1), rel_center],
            dim=-1,
        )
        return pooled


class RPDiffStyleRelationEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        out_dim=256,
        num_heads=4,
        use_lang_emb=True,
        lang_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        self.use_lang_emb = use_lang_emb
        self.static_branch = RelationBranch(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.candidate_branch = RelationBranch(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.time_emb = nn.Sequential(
            SinusoidalTimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pose_emb = nn.Sequential(
            nn.Linear(9, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        if use_lang_emb:
            self.lang_proj = nn.Sequential(
                nn.Linear(lang_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )
        fusion_dim = self.static_branch.out_dim + self.candidate_branch.out_dim + embed_dim * 2
        if use_lang_emb:
            fusion_dim += embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _squeeze_pc(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"{name} expected [B,N,3] or [B,1,N,3], got {tuple(x.shape)}")
        return x

    def _unnormalize_goal(self, noisy_goal_pose9d_norm: torch.Tensor, normalizer) -> torch.Tensor:
        if normalizer is None:
            raise RuntimeError("RPDiffStyleRelationEncoder requires a fitted normalizer before forward().")
        if hasattr(normalizer, "__getitem__"):
            return normalizer["goal_pose9d"].unnormalize(noisy_goal_pose9d_norm)
        return normalizer.unnormalize(noisy_goal_pose9d_norm)

    def forward(
        self,
        noisy_goal_pose9d_norm: torch.Tensor,
        timestep: torch.Tensor,
        pc_manipulated: torch.Tensor,
        pc_target: torch.Tensor,
        normalizer,
        lang_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pc_manipulated = self._squeeze_pc(pc_manipulated, "pc_manipulated")
        pc_target = self._squeeze_pc(pc_target, "pc_target")
        pose9d = self._unnormalize_goal(noisy_goal_pose9d_norm, normalizer)
        T_t = pose9d_to_matrix(pose9d)
        pc_candidate = transform_point_cloud(pc_manipulated, T_t)

        static_feat = self.static_branch(pc_manipulated, pc_target)
        candidate_feat = self.candidate_branch(pc_candidate, pc_target)
        t_feat = self.time_emb(timestep.reshape(-1).to(pc_manipulated.device))
        pose_feat = self.pose_emb(noisy_goal_pose9d_norm)
        feats = [static_feat, candidate_feat, t_feat, pose_feat]

        if self.use_lang_emb and lang_emb is not None:
            if lang_emb.ndim == 3:
                lang_emb = lang_emb.squeeze(1)
            feats.append(self.lang_proj(lang_emb.float()))

        return self.fusion(torch.cat(feats, dim=-1))
