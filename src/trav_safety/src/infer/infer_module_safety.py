import torch
import torch.nn as nn

from ..models.WSTNet_safety_bev import TravNet
# from models.WSTNet_safety_final2 import TravNet

class InferModuleSafety(nn.Module):
    """
    推理模块，适配WSTNet_safety_bev和WSTNet_safety_final2
    """
    def __init__(self, configs):
        super().__init__()
        self.model = TravNet(
            configs.MODEL.GRID_BOUNDS,
            configs.MODEL.INPUT_SIZE,
            downsample=configs.MODEL.DOWNSAMPLE,
            image_dim=configs.MODEL.LATENT_DIM,
            temporal_length=configs.MODEL.TIME_LENGTH,
            predict_depth=configs.MODEL.PREDICT_DEPTH,
            fuse_pcloud=configs.MODEL.FUSE_PCLOUD
        )

    def forward(self, color_img, pcloud, intrinsics, extrinsics, depth_target=None):
        """
        Args:
            color_img (torch.Tensor): 输入彩色图像
            pcloud (torch.Tensor): 点云体素
            intrinsics (torch.Tensor): 相机内参
            extrinsics (torch.Tensor): 相机外参
            depth_target (torch.Tensor, optional): 深度目标

        Returns:
            trav_map (torch.Tensor): 可通过性预测
            safety_map (torch.Tensor): 安全系数预测
            depth_logits (torch.Tensor): 深度预测
            debug (torch.Tensor): 调试信息
        """
        trav_map, safety_map, depth_logits, debug = self.model(
            color_img, pcloud, intrinsics, extrinsics, depth_target
        )
        return trav_map, safety_map, depth_logits, debug
