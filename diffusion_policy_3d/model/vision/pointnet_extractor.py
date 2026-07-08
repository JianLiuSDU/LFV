import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 use_lang_emb=False,
                 use_stage_emb=False,
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        
        # 1. 动态检测：YAML里有没有配 point_cloud
        self.use_point_cloud = self.point_cloud_key in observation_space

        # 动态计算最终输出给 U-Net 的特征总维度
        self.n_output_channels = 0

        # ==========================================
        # A. 状态(位姿)特征提取 (MVP核心，永远保留)
        # ==========================================
        self.state_shape = observation_space[self.state_key]
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        self.n_output_channels += output_dim

        # ==========================================
        # B. 3D 点云特征提取 (动态屏蔽)
        # ==========================================
        self.pointcloud_encoder_cfg = pointcloud_encoder_cfg
        if self.use_point_cloud:
            cprint("[DP3Encoder] 视觉模式：检测到点云，初始化 PointNet", "green")
            self.point_cloud_shape = observation_space[self.point_cloud_key]
            if pointnet_type == "pointnet":
                if use_pc_color:
                    pointcloud_encoder_cfg.in_channels = 6
                    self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                else:
                    pointcloud_encoder_cfg.in_channels = 3
                    self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            self.n_output_channels += out_channel
        else:
            cprint("[DP3Encoder] MVP模式：未配置点云，视觉网络已自动屏蔽", "yellow")

        # ==========================================
        # C. 语言和任务阶段特征 (动态屏蔽)
        # ==========================================
        self.use_lang_emb = use_lang_emb
        if self.use_lang_emb:
            self.lang_preprocess = nn.Linear(1024, 256 * 2)
            self.n_output_channels += 256 * 2
            self.lang_key = "lang_token_embs"

        self.use_stage_emb = use_stage_emb
        if self.use_stage_emb:
            self.n_output_channels += 3
            self.stage_key = "stage_embs"

        cprint(f"[DP3Encoder] use_lang_emb: {self.use_lang_emb}", "cyan")
        cprint(f"[DP3Encoder] use_stage_emb: {self.use_stage_emb}", "cyan")
        cprint(f"[DP3Encoder] 最终输出特征维度: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:
        # 1. 提取基础的位姿状态特征 (始终执行)
        state = observations[self.state_key]
        final_feat = self.state_mlp(state)  # B * 64

        # 2. 如果启用了点云，提取点云特征并拼接到前面
        if getattr(self, 'use_point_cloud', False):
            points = observations[self.point_cloud_key]
            pn_feat = self.extractor(points)
            final_feat = torch.cat([pn_feat, final_feat], dim=-1)

        # 3. 如果启用了语言或阶段，提取并拼接到前面
        if getattr(self, 'use_lang_emb', False) or getattr(self, 'use_stage_emb', False):
            emb_list = []
            if self.use_lang_emb:
                emb_list.append(self.lang_preprocess(observations[self.lang_key]))
            if self.use_stage_emb:
                emb_list.append(observations[self.stage_key])
            
            emb_feat = torch.cat(emb_list, dim=-1)
            n_repeat = len(state) // len(emb_feat)
            emb_feat = emb_feat.repeat(n_repeat, 1)

            final_feat = torch.cat((emb_feat, final_feat), dim=-1)

        return final_feat

    def output_shape(self):
        return self.n_output_channels