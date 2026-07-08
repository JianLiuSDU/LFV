# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimplePointNet(nn.Module):
#     """
#     轻量级 PointNet 骨干网络
#     专门针对 N=256 的稀疏点云设计，提取逐点局部特征
#     """
#     def __init__(self, in_channels=3, out_channels=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels, 64, 1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
            
#             nn.Conv1d(64, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
            
#             nn.Conv1d(128, out_channels, 1),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.net(x)


# class CrossAttentionDP3Encoder(nn.Module):
#     """
#     基于双点云交叉注意力与多模态投影融合的 DP3 特征编码器
#     """
#     def __init__(self, 
#                  observation_space: dict, 
#                  out_channel=256,   # 💡 建议将输出特征提至 256，防止最终瓶颈
#                  use_lang_emb=True, 
#                  lang_dim=1024):
#         super().__init__()
#         self._out_channel = out_channel
#         self.use_lang_emb = use_lang_emb
        
#         # --- 核心特征维度定义 ---
#         self.d_pc = 256        # 视觉：单个点云特征维度
#         self.d_lang = 256      # 文本：降维后的语言特征
#         self.d_pos = 64        # 状态：升维后的机械臂位姿特征
        
#         # 1. 视觉：共享权重的 PointNet 骨干
#         self.point_backbone = SimplePointNet(in_channels=3, out_channels=self.d_pc)
        
#         # 可学习的“身份区分嵌入” (Type Embedding)
#         self.man_type_emb = nn.Parameter(torch.randn(1, 1, self.d_pc) * 0.02)
#         self.tgt_type_emb = nn.Parameter(torch.randn(1, 1, self.d_pc) * 0.02)
        
#         # 交叉注意力机制
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.d_pc, 
#             num_heads=4, 
#             dropout=0.1, 
#             batch_first=True
#         )
        
#         # 2. 状态：agent_pos 升维网络 (放大话语权)
#         self.agent_pos_dim = observation_space['agent_pos'][0] 
#         self.pos_proj = nn.Sequential(
#             nn.Linear(self.agent_pos_dim, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, self.d_pos)
#         )
        
#         # 3. 文本：语言特征降维网络 (压缩冗余，防止特征淹没)
#         if self.use_lang_emb:
#             self.lang_proj = nn.Sequential(
#                 nn.Linear(lang_dim, 512),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(512, self.d_lang)
#             )
        
#         # 4. 最终融合网络
#         concat_dim = self.d_pc * 2 + self.d_pos
#         if self.use_lang_emb:
#             concat_dim += self.d_lang
            
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(concat_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, self._out_channel)
#         )

#     def forward(self, obs_dict):
#         # 1. 获取并转置局部点云
#         pc_man = obs_dict['pc_manipulated'].transpose(1, 2)
#         pc_tgt = obs_dict['pc_target'].transpose(1, 2)
        
#         # 2. 提取视觉特征
#         feat_man_pts = self.point_backbone(pc_man).transpose(1, 2)
#         feat_tgt_pts = self.point_backbone(pc_tgt).transpose(1, 2)
        
#         # 3. 注入身份特征
#         feat_man_pts = feat_man_pts + self.man_type_emb
#         feat_tgt_pts = feat_tgt_pts + self.tgt_type_emb
        
#         # 4. Cross-Attention
#         attn_out, _ = self.cross_attn(
#             query=feat_man_pts, 
#             key=feat_tgt_pts, 
#             value=feat_tgt_pts
#         )
        
#         # 5. 全局最大池化
#         global_man = torch.max(feat_man_pts, dim=1)[0]
#         global_cross = torch.max(attn_out, dim=1)[0]
        
#         # 6. 处理状态特征 (投影升维)
#         agent_pos = obs_dict['agent_pos']
#         pos_feat = self.pos_proj(agent_pos)
        
#         features = [global_man, global_cross, pos_feat]
        
#         # 7. 处理语言特征 (投影降维 + 动态时序对齐)
#         if self.use_lang_emb and 'lang_token_embs' in obs_dict:
#             lang_emb = obs_dict['lang_token_embs']
            
#             # 展平多余维度 [B, 1, 1024] -> [B, 1024]
#             if len(lang_emb.shape) == 3:  
#                 lang_emb = lang_emb.squeeze(1)
                
#             # 时序对齐防御：当 obs_horizon > 1 时，复制文本特征以对齐 Batch*Time
#             B_times_T = agent_pos.shape[0] 
#             B_lang = lang_emb.shape[0]     
#             if B_lang != B_times_T:
#                 time_steps = B_times_T // B_lang
#                 lang_emb = lang_emb.repeat_interleave(time_steps, dim=0)
                
#             # 投影降维
#             lang_feat = self.lang_proj(lang_emb)
#             features.append(lang_feat)
            
#         # 8. 特征拼接与输出
#         concat_feat = torch.cat(features, dim=-1)
#         out_feat = self.fusion_mlp(concat_feat)
        
#         return out_feat

#     def output_shape(self):
#         return self._out_channel
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. 3D Fourier Positional Encoding
# =========================
class FourierPositionalEncoding3D(nn.Module):
    def __init__(self, num_bands=6, include_input=True):
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input
        freq_bands = 2.0 ** torch.arange(num_bands) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self):
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_bands

    def forward(self, xyz):
        # xyz: [..., 3]
        x = xyz.unsqueeze(-1) * self.freq_bands
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        enc = torch.cat([sin_x, cos_x], dim=-1)   # [..., 3, 2F]
        enc = enc.flatten(start_dim=-2)           # [..., 3*2F]
        if self.include_input:
            enc = torch.cat([xyz, enc], dim=-1)
        return enc


# =========================
# 2. Lightweight Point Backbone
# =========================
class PointMLP(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, N, C]
        return self.net(x.transpose(1, 2))  # [B, C, N]


# =========================
# 3. Role-specific adapter
# =========================
class RoleAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 4. Cross-object kNN relation
# =========================
class CrossKNNRelation(nn.Module):
    """
    让 manipulated 点只看 target 中最近的 k 个点
    更符合接触/支撑/靠近关系建模
    """
    def __init__(self, feat_dim, rel_pe_dim, k=8):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_dim * 2 + rel_pe_dim + 1, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, src_xyz, src_feat, dst_xyz, dst_feat, pos3d):
        """
        src_xyz:  [B, Ns, 3]
        src_feat: [B, C, Ns]
        dst_xyz:  [B, Nd, 3]
        dst_feat: [B, C, Nd]
        """
        B, Ns, _ = src_xyz.shape
        Nd = dst_xyz.shape[1]
        C = src_feat.shape[1]
        k = min(self.k, Nd)

        # pairwise distance
        dists = torch.cdist(src_xyz, dst_xyz)              # [B, Ns, Nd]
        knn_dists, knn_idx = torch.topk(dists, k=k, dim=-1, largest=False)  # [B, Ns, k]

        # gather dst xyz
        dst_xyz_expand = dst_xyz.unsqueeze(1).expand(-1, Ns, -1, -1)         # [B, Ns, Nd, 3]
        idx_xyz = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)                # [B, Ns, k, 3]
        dst_xyz_knn = torch.gather(dst_xyz_expand, 2, idx_xyz)               # [B, Ns, k, 3]

        # gather dst feat
        dst_feat_expand = dst_feat.unsqueeze(2).expand(-1, -1, Ns, -1)       # [B, C, Ns, Nd]
        idx_feat = knn_idx.unsqueeze(1).expand(-1, C, -1, -1)                # [B, C, Ns, k]
        dst_feat_knn = torch.gather(dst_feat_expand, 3, idx_feat)            # [B, C, Ns, k]

        # src feat repeat
        src_feat_expand = src_feat.unsqueeze(-1).expand(-1, -1, -1, k)       # [B, C, Ns, k]

        # relative geometry
        src_xyz_expand = src_xyz.unsqueeze(2).expand(-1, -1, k, -1)          # [B, Ns, k, 3]
        rel_xyz = src_xyz_expand - dst_xyz_knn                                # [B, Ns, k, 3]
        rel_pe = pos3d(rel_xyz).permute(0, 3, 1, 2).contiguous()             # [B, Cr, Ns, k]

        knn_dists = knn_dists.unsqueeze(1)                                    # [B, 1, Ns, k]

        pair_feat = torch.cat([src_feat_expand, dst_feat_knn, rel_pe, knn_dists], dim=1)
        pair_feat = self.mlp(pair_feat)                                       # [B, C, Ns, k]

        # 聚合 target 近邻关系到每个 manipulated 点
        rel_feat = torch.max(pair_feat, dim=-1)[0]                            # [B, C, Ns]
        return rel_feat, dists


# =========================
# 5. Main Encoder
# =========================
class ManipulationCentricSE3Encoder(nn.Module):
    """
    面向 SE(3) 轨迹扩散的点云条件编码器

    核心特点：
    1) manipulated-centric（不再完全对称）
    2) 同时保留 task-local absolute coords 和 self-centered coords
    3) 用 cross-object kNN relation 捕捉接触/支撑/靠近关系
    4) 加入稳定的显式几何摘要
    5) 提取 target relevant region，而不是只做 whole target pooling
    """
    def __init__(self,
                 observation_space: dict,
                 out_channel=256,
                 use_lang_emb=True,
                 lang_dim=1024,
                 d_pc=128,
                 d_lang=128,
                 d_pos=64,
                 d_geom=64,
                 pe_bands=6,
                 k_rel=8):
        super().__init__()
        self._out_channel = out_channel
        self.use_lang_emb = use_lang_emb
        self.d_pc = d_pc
        self.d_lang = d_lang
        self.d_pos = d_pos

        # 3D PE
        self.pos3d = FourierPositionalEncoding3D(num_bands=pe_bands, include_input=True)
        pe_dim = self.pos3d.out_dim

        # 每个点输入:
        # abs PE + self-centered PE + other-centered PE
        point_in_dim = pe_dim * 3

        # shared stem + role-specific adapters
        self.shared_stem = PointMLP(in_channels=point_in_dim, out_channels=d_pc)
        self.man_adapter = RoleAdapter(d_pc)
        self.tgt_adapter = RoleAdapter(d_pc)

        # type embeddings
        self.man_type_emb = nn.Parameter(torch.randn(1, d_pc, 1) * 0.02)
        self.tgt_type_emb = nn.Parameter(torch.randn(1, d_pc, 1) * 0.02)

        # cross relation
        self.cross_rel = CrossKNNRelation(
            feat_dim=d_pc,
            rel_pe_dim=pe_dim,
            k=k_rel
        )

        # target relevant region pooling temperature
        self.region_scale = nn.Parameter(torch.tensor(10.0))

        # geometry summary:
        # c_man(3), c_tgt(3), delta(3), dist(1), unit_delta(3),
        # cov_man(6), cov_tgt(6), contact_stats(4)
        geom_in_dim = 3 + 3 + 3 + 1 + 3 + 6 + 6 + 4
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_geom)
        )

        # agent pose
        self.agent_pos_dim = observation_space['agent_pos'][0]
        self.pos_proj = nn.Sequential(
            nn.Linear(self.agent_pos_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_pos)
        )

        # language
        if self.use_lang_emb:
            self.lang_proj = nn.Sequential(
                nn.Linear(lang_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, d_lang)
            )

        # fusion
        # global_man(2C) + global_rel(2C) + global_tgt(2C) + region_tgt(C) + geom + pos (+ lang)
        concat_dim = 2*d_pc + 2*d_pc + 2*d_pc + d_pc + d_geom + d_pos
        if self.use_lang_emb:
            concat_dim += d_lang

        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, out_channel)
        )

    # ---------- helper ----------
    def pool_global(self, feat):
        # feat: [B, C, N]
        max_feat = torch.max(feat, dim=-1)[0]
        mean_feat = torch.mean(feat, dim=-1)
        return torch.cat([max_feat, mean_feat], dim=-1)

    def flatten_cov6(self, x_centered):
        """
        x_centered: [B, N, 3]
        return: [B, 6]
        """
        B, N, _ = x_centered.shape
        cov = torch.matmul(x_centered.transpose(1, 2), x_centered) / max(N, 1)   # [B, 3, 3]
        return torch.stack([
            cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2],
            cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2]
        ], dim=-1)

    def build_point_input(self, xyz, c_self, c_other):
        abs_pe = self.pos3d(xyz)
        self_pe = self.pos3d(xyz - c_self)
        other_pe = self.pos3d(xyz - c_other)
        return torch.cat([abs_pe, self_pe, other_pe], dim=-1)

    def encode_agent_pos(self, agent_pos):
        # 兼容 [B, D] / [B, 1, D]
        if agent_pos.dim() == 3 and agent_pos.shape[1] == 1:
            agent_pos = agent_pos.squeeze(1)
        elif agent_pos.dim() > 2:
            agent_pos = agent_pos.reshape(agent_pos.shape[0], -1)
        return self.pos_proj(agent_pos)

    # ---------- main ----------
    def forward(self, obs_dict):
        """
        obs_dict:
            pc_manipulated: [B, N, 3]
            pc_target:      [B, N, 3]
            agent_pos:      [B, D] or [B, 1, D]
            lang_token_embs: optional
        """
        pc_man = obs_dict['pc_manipulated']
        pc_tgt = obs_dict['pc_target']

        # 如果上游送的是 [B, 1, N, 3]，这里自动 squeeze
        if pc_man.dim() == 4 and pc_man.shape[1] == 1:
            pc_man = pc_man.squeeze(1)
        if pc_tgt.dim() == 4 and pc_tgt.shape[1] == 1:
            pc_tgt = pc_tgt.squeeze(1)

        # ----------------------------------
        # 1. object-level geometry
        # ----------------------------------
        c_man = pc_man.mean(dim=1, keepdim=True)     # [B, 1, 3]
        c_tgt = pc_tgt.mean(dim=1, keepdim=True)     # [B, 1, 3]
        delta = (c_tgt - c_man).squeeze(1)           # [B, 3]
        dist = torch.norm(delta, dim=-1, keepdim=True)  # [B, 1]
        unit_delta = delta / (dist + 1e-6)

        man_centered = pc_man - c_man
        tgt_centered = pc_tgt - c_tgt

        # ----------------------------------
        # 2. pointwise inputs
        # ----------------------------------
        man_in = self.build_point_input(pc_man, c_man, c_tgt)   # [B, N, Cin]
        tgt_in = self.build_point_input(pc_tgt, c_tgt, c_man)   # [B, N, Cin]

        # ----------------------------------
        # 3. self feature stems
        # ----------------------------------
        feat_man = self.shared_stem(man_in)
        feat_tgt = self.shared_stem(tgt_in)

        feat_man = self.man_adapter(feat_man) + self.man_type_emb
        feat_tgt = self.tgt_adapter(feat_tgt) + self.tgt_type_emb

        # ----------------------------------
        # 4. cross-object local relation
        # manipulated points only look at k nearest target points
        # ----------------------------------
        rel_man, dists_full = self.cross_rel(
            src_xyz=pc_man,
            src_feat=feat_man,
            dst_xyz=pc_tgt,
            dst_feat=feat_tgt,
            pos3d=self.pos3d
        )
        feat_man_rel = feat_man + rel_man

        # ----------------------------------
        # 5. target relevant region token
        # 不是整个 target 一锅端，而是提取与 manipulated 当前最相关区域
        # ----------------------------------
        # dists_full: [B, Nm, Nt]
        d_tgt_to_man = torch.min(dists_full, dim=1)[0]   # [B, Nt]
        region_w = F.softmax(-self.region_scale * d_tgt_to_man, dim=-1)  # [B, Nt]
        region_tgt = torch.sum(feat_tgt * region_w.unsqueeze(1), dim=-1) # [B, C]

        # ----------------------------------
        # 6. explicit geometry summary
        # ----------------------------------
        cov_man = self.flatten_cov6(man_centered)
        cov_tgt = self.flatten_cov6(tgt_centered)

        d_man_to_tgt = torch.min(dists_full, dim=-1)[0]  # [B, Nm]
        contact_stats = torch.stack([
            torch.min(d_man_to_tgt, dim=-1)[0],
            torch.mean(d_man_to_tgt, dim=-1),
            torch.max(d_man_to_tgt, dim=-1)[0],
            torch.std(d_man_to_tgt, dim=-1, unbiased=False)
        ], dim=-1)  # [B, 4]

        geom_feat = torch.cat([
            c_man.squeeze(1),   # 3
            c_tgt.squeeze(1),   # 3
            delta,              # 3
            dist,               # 1
            unit_delta,         # 3
            cov_man,            # 6
            cov_tgt,            # 6
            contact_stats       # 4
        ], dim=-1)
        geom_feat = self.geom_proj(geom_feat)

        # ----------------------------------
        # 7. global tokens
        # ----------------------------------
        global_man = self.pool_global(feat_man)           # [B, 2C]
        global_rel = self.pool_global(feat_man_rel)       # [B, 2C]
        global_tgt = self.pool_global(feat_tgt)           # [B, 2C]

        # ----------------------------------
        # 8. agent_pos
        # ----------------------------------
        pos_feat = self.encode_agent_pos(obs_dict['agent_pos'])

        features = [global_man, global_rel, global_tgt, region_tgt, geom_feat, pos_feat]

        # ----------------------------------
        # 9. language
        # ----------------------------------
        if self.use_lang_emb and 'lang_token_embs' in obs_dict:
            lang_emb = obs_dict['lang_token_embs']

            if lang_emb.dim() == 3 and lang_emb.shape[1] == 1:
                lang_emb = lang_emb.squeeze(1)

            if lang_emb.dim() > 2:
                lang_emb = lang_emb.reshape(lang_emb.shape[0], -1)

            B_obs = pos_feat.shape[0]
            B_lang = lang_emb.shape[0]
            if B_lang != B_obs:
                repeat_factor = B_obs // B_lang
                lang_emb = lang_emb.repeat_interleave(repeat_factor, dim=0)

            lang_feat = self.lang_proj(lang_emb)
            features.append(lang_feat)

        # ----------------------------------
        # 10. fuse
        # ----------------------------------
        concat_feat = torch.cat(features, dim=-1)
        out_feat = self.fusion_mlp(concat_feat)
        return out_feat

    def output_shape(self):
        return self._out_channel


class LearnedQueryPooling(nn.Module):
    def __init__(self, d_model=128, num_queries=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        B = tokens.shape[0]
        queries = self.queries.expand(B, -1, -1)
        pooled, _ = self.attn(query=queries, key=tokens, value=tokens, need_weights=False)
        pooled = self.norm(pooled + queries)
        pooled = self.ffn_norm(pooled + self.ffn(pooled))
        return pooled


class GoalConditionedSetTransformerEncoder(nn.Module):
    """
    Goal-conditioned set transformer encoder for full64 residual trajectory diffusion.

    Stage A:
        local point clouds -> FourierPE -> shared PointMLP -> role embedding
        -> per-object learned query pooling

    Stage B:
        CLS/start/goal/lang/pooled point tokens -> TransformerEncoder -> CLS -> MLP.
    """

    def __init__(
        self,
        observation_space: dict,
        out_channel=256,
        use_lang_emb=True,
        lang_dim=1024,
        d_model=128,
        num_heads=4,
        num_layers=3,
        ffn_dim=512,
        dropout=0.1,
        fourier_bands=6,
        use_goal_abs_token=True,
        token_pooling=None,
    ):
        super().__init__()
        self._out_channel = out_channel
        self.use_lang_emb = use_lang_emb
        self.use_goal_abs_token = use_goal_abs_token
        self.d_model = d_model

        token_pooling = token_pooling or {}
        pooling_type = token_pooling.get("type", "pma")
        if pooling_type not in ("pma", "learned_query_attention"):
            raise ValueError(f"Unsupported token_pooling.type: {pooling_type}")
        k_man = int(token_pooling.get("k_man", 16))
        k_target = int(token_pooling.get("k_target", 16))
        pool_heads = int(token_pooling.get("num_heads", num_heads))
        pool_dropout = float(token_pooling.get("dropout", dropout))

        self.pos3d = FourierPositionalEncoding3D(num_bands=fourier_bands, include_input=True)
        pe_dim = self.pos3d.out_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(pe_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.man_role_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_role_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.man_pool = LearnedQueryPooling(d_model, k_man, pool_heads, pool_dropout)
        self.target_pool = LearnedQueryPooling(d_model, k_target, pool_heads, pool_dropout)

        agent_pos_dim = observation_space["agent_pos"][0]
        self.start_pose_mlp = nn.Sequential(
            nn.Linear(agent_pos_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.start_role_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.goal_delta_mlp = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.goal_delta_role_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if self.use_goal_abs_token:
            self.goal_abs_mlp = nn.Sequential(
                nn.Linear(9, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.goal_abs_role_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if self.use_lang_emb:
            self.lang_mlp = nn.Sequential(
                nn.Linear(lang_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.lang_role_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_channel),
        )

        self.debug_shapes = {}

    def _squeeze_time(self, x, last_dim):
        if x.dim() == 3 and x.shape[1] == 1 and x.shape[-1] == last_dim:
            return x.squeeze(1)
        if x.dim() > 2 and x.shape[-1] == last_dim:
            return x.reshape(x.shape[0], -1, last_dim)[:, -1, :]
        return x

    def _squeeze_points(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    def _encode_points(self, pc, role_emb, pool):
        pe = self.pos3d(pc)
        raw_tokens = self.point_mlp(pe) + role_emb
        pooled_tokens = pool(raw_tokens)
        return raw_tokens, pooled_tokens

    def _pose_token(self, x, mlp, role_emb, dim):
        x = self._squeeze_time(x, dim)
        token = mlp(x).unsqueeze(1) + role_emb
        return token

    def forward(self, obs_dict):
        pc_man = self._squeeze_points(obs_dict["pc_manipulated"])
        pc_tgt = self._squeeze_points(obs_dict["pc_target"])
        B = pc_man.shape[0]

        raw_man, pooled_man = self._encode_points(pc_man, self.man_role_emb, self.man_pool)
        raw_tgt, pooled_tgt = self._encode_points(pc_tgt, self.target_role_emb, self.target_pool)

        start_token = self._pose_token(
            obs_dict["agent_pos"], self.start_pose_mlp, self.start_role_emb, 7
        )
        goal_delta_token = self._pose_token(
            obs_dict["goal_delta_pose9d"], self.goal_delta_mlp, self.goal_delta_role_emb, 9
        )

        tokens = [
            self.cls_token.expand(B, -1, -1),
            start_token,
            goal_delta_token,
        ]

        if self.use_goal_abs_token:
            if "goal_pose9d" not in obs_dict:
                raise KeyError("GoalConditionedSetTransformerEncoder requires obs['goal_pose9d']")
            tokens.append(self._pose_token(
                obs_dict["goal_pose9d"], self.goal_abs_mlp, self.goal_abs_role_emb, 9
            ))

        if self.use_lang_emb:
            if "lang_token_embs" in obs_dict:
                lang = obs_dict["lang_token_embs"]
                if lang.dim() == 3 and lang.shape[1] == 1:
                    lang = lang.squeeze(1)
                elif lang.dim() > 2:
                    lang = lang.reshape(lang.shape[0], -1)
            else:
                lang = torch.zeros(B, 1024, device=pc_man.device, dtype=pc_man.dtype)
            if lang.shape[0] != B:
                repeat_factor = B // lang.shape[0]
                lang = lang.repeat_interleave(repeat_factor, dim=0)
            tokens.append(self.lang_mlp(lang).unsqueeze(1) + self.lang_role_emb)

        tokens.extend([pooled_man, pooled_tgt])
        fusion_tokens = torch.cat(tokens, dim=1)
        encoded = self.transformer(fusion_tokens)
        cls_feat = encoded[:, 0]
        out = self.fusion_mlp(cls_feat)

        self.debug_shapes = {
            "raw_man_tokens": tuple(raw_man.shape),
            "pooled_man_tokens": tuple(pooled_man.shape),
            "raw_target_tokens": tuple(raw_tgt.shape),
            "pooled_target_tokens": tuple(pooled_tgt.shape),
            "fusion_tokens": tuple(fusion_tokens.shape),
            "global_cond": tuple(out.shape),
        }
        return out

    def output_shape(self):
        return self._out_channel
