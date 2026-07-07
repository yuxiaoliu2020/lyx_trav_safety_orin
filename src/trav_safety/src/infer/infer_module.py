import torch
import torch.nn as nn
# import pytorch_lightning as pl
# from models.traversability_net_wtconv import TravNet
from models.TravNet import TravNet
# from models.STANet import TravNet

class InferModule(nn.Module):
    """
    用PyTorch Lightning实现的TravNet模型的推理模块
    """
    def __init__(self, configs):
        """
        初始化推理模块

        Args:
            configs (object): 包含推理所需的各种参数
        """
        super().__init__()

        # 创建TravNet模型
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
        前向传播方法，用于推理

        Args:
            color_img (torch.Tensor): 输入的彩色图像
            pcloud (torch.Tensor): 点云数据
            intrinsics (torch.Tensor): 相机内参
            extrinsics (torch.Tensor): 相机外参
            depth_target (torch.Tensor, optional): 深度目标（用于训练时计算损失）

        Returns:
            trav_map (torch.Tensor): 通过性地图
            pred_depth (torch.Tensor): 深度预测
            debug (torch.Tensor): 调试信息
        """
        trav_map, pred_depth, debug = self.model(color_img, pcloud, intrinsics, extrinsics, depth_target)
        return trav_map, pred_depth, debug