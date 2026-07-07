# 4月15日18:23改：wtconv_refine层后添加点卷积层、bev_compressor中wt_levels从1改为2、wtconv_refine中wt_levels从2改为3
# (bev_compressor中不添加点卷积层是因为WTConv直接在高维特征上操作会导致参数激增，并且后面的其他层已经提供了通道间信息交流)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_fusion_wtconv import TemporalModel
from .encoder_decoder import Encoder, Decoder
from .wtconv.wtconv2d import WTConv2d


'''
# temporal_model.yaml

TAG: 'temporal'

DATASET:
    TRAIN_DATA: ['dataset/zed2/data_train', 'dataset/realsense/data_train']
    VALID_DATA: ['dataset/zed2/data_valid', 'dataset/realsense/data_valid']

OPTIMIZER:
    LR: 0.0001
    WEIGHT_DECAY: 0.0001

TRAINING:
    BATCHSIZE: 4
    WORKERS: 4
    EPOCHS: 20
    VERBOSE: False                      # 是否打印训练信息
    GAMMA: 1.0
    DEPTH_WEIGHT: 0.1
    HORIZON: 300                        # 时间窗口长度
    DT: 0.1                             # 采样时间间隔

MODEL:
    TIME_LENGTH: 6                      # 输入batch包含的图片帧数
    DOWNSAMPLE: 8                       # 下采样倍数
    LATENT_DIM: 64
    INPUT_SIZE: [320, 180]              # 输入图像尺寸
    PREDICT_DEPTH: True                 # 是否预测深度
    TRAIN_DEPTH: True                   # 是否训练深度
    FUSE_PCLOUD: True                   # 是否融合点云
    GRID_BOUNDS: {
        'xbound': [-2.0, 8.0, 0.1],
        'ybound': [-5.0, 5.0, 0.1],
        'zbound': [-1.0, 2.0, 0.2],
        'dbound': [ 0.3, 8.0, 0.2]}     # 网格边界

AUGMENTATIONS:
    MAX_TRANSLATION: 0.0                # 最大平移
    MAX_ROTATION: 0.0                   # 最大旋转
'''

class TravNet(nn.Module):
    """
    TravNet: 一个预测可通过性和路径导航的神经网络

    Attributes:
        grid_bounds (dict): 体素网格的边界
        input_size (tuple): 输入图像尺寸
        downsample (int): 下采样因子，只能为8或4
        image_dim (int): 图像维度
        temporal_length (int): 时间序列的长度
        predict_depth (bool): 是否预测深度
        fuse_pcloud (bool): 是否融合点云
    """
    def __init__(
            self, grid_bounds, input_size, downsample=8,
            image_dim=64, temporal_length=6, predict_depth=True,
            fuse_pcloud=True
        ):
        
        super(TravNet, self).__init__()

        self.grid_bounds    = grid_bounds
        self.input_size     = (input_size[1], input_size[0]) # 从(H, W)换到(W, H)
        self.camC           = image_dim
        self.downsample     = downsample
        self.predict_depth  = predict_depth
        self.fuse_pcloud    = fuse_pcloud
        self.eps            = 1e-6

        dx = torch.Tensor([row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        
        self.int_nx = nx.cpu().detach().numpy()

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()

        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

        self.voxels = self.create_voxels()

        if self.predict_depth:
            self.D = int((grid_bounds['dbound'][1] - grid_bounds['dbound'][0]) / grid_bounds['dbound'][2])
        else:
            self.D = 0

        self.latent_dim = image_dim

        if self.fuse_pcloud:
            pcloud_dim = self.nx[2]
        else:
            pcloud_dim = 0
        
        self.encoder = Encoder(self.D + self.camC, downsample=self.downsample)
        
        self.bev_compressor = nn.Sequential(
            # nn.Conv2d(self.camC * self.nx[2] + pcloud_dim, self.camC, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(self.camC * self.nx[2] + pcloud_dim, self.camC, kernel_size=1, bias=False),
            WTConv2d(self.camC, self.camC, kernel_size=3, wt_levels=2),
            nn.BatchNorm2d(self.camC),
            nn.ReLU(inplace=True))
        
        self.temporal_model = TemporalModel(
            channels=self.camC,
            temporal_length=temporal_length,
            input_shape=(nx[0], nx[1]))

        # self.temporal_model = TemporalModel(
        # channels=self.camC,
        # temporal_length=temporal_length,
        # input_shape=(nx[0], nx[1]),
        # window_size=3,
        # num_heads=4
        # )

        self.decoder = Decoder(in_channels=self.latent_dim)

        # 在Decoder之后、travmap_head之前添加WTConv2d处理层
        self.wtconv_refine = WTConv2d(
            in_channels=self.latent_dim,
            out_channels=self.latent_dim,
            kernel_size=3,
            # wt_levels=1,
            wt_levels=3,
            wt_type='db2'  # 使用Daubechies-2小波
        )

        self.features_refine = nn.Sequential(
            self.wtconv_refine,
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=1), # 点卷积
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)   # 若过拟合，调整至0.3-0.5
        )
        
        self.travmap_head = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
            # WTConv2d(self.latent_dim, self.latent_dim, kernel_size=3, wt_levels=1),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1), # 若过拟合，调整至0.2
            nn.Conv2d(self.latent_dim, 2, kernel_size=1, padding=0),
            nn.Sigmoid())
    
    def create_voxels(self):
        grid_z = torch.arange(*self.grid_bounds['zbound'], dtype=torch.float).flip(0)
        grid_z = torch.reshape(grid_z, [self.nx[2], 1, 1])
        grid_z = grid_z.repeat(1, self.nx[0], self.nx[1])

        grid_y = torch.arange(*self.grid_bounds['ybound'], dtype=torch.float).flip(0)
        grid_y = torch.reshape(grid_y, [1, 1, self.nx[1]])
        grid_y = grid_y.repeat(self.nx[2], self.nx[0], 1)

        grid_x = torch.arange(*self.grid_bounds['xbound'], dtype=torch.float).flip(0)
        grid_x = torch.reshape(grid_x, [1, self.nx[0], 1])
        grid_x = grid_x.repeat(self.nx[2], 1, self.nx[1])

        voxels = torch.stack((grid_x, grid_y, grid_z), -1)
        return nn.Parameter(voxels, requires_grad=False)
    
    def get_inv_geometry(self, intrinsics, extrinsics):
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        points = self.voxels.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = points.expand(B, -1, -1, -1, -1, -1, -1)

        points = points - translation.view(B, N, 1, 1, 1, 3, 1)
        combined_transformation = intrinsics.matmul(torch.inverse(rotation))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        points = torch.cat((points[..., :2] / (points[..., 2:3] + self.eps), points[..., 2:3]), -1)

        return points
    
    def sample2bev(self, geometry, x):
        _, d, h, w, c = x.shape
        batch, T, Z, X, Y, _ = geometry.shape
        geometry = geometry.view(batch*T, Z, X, Y, 3)

        u = 2 * geometry[..., 0] / (self.input_size[1]-1) - 1
        v = 2 * geometry[..., 1] / (self.input_size[0]-1) - 1
        depth = 2 * (geometry[...,2] - self.grid_bounds['dbound'][0]) / (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0]) - 1
        
        grid = torch.stack((u, v, depth), -1)

        x = F.grid_sample(x, grid, align_corners=False)
        
        x = x.view(batch, T, *x.shape[1:])
        return x
        
    def forward(self, color_img, pcloud, intrinsics, extrinsics, depth_img=None):
        B, T, C, imH, imW = color_img.shape
        assert(C==3)
        color_img = color_img.view(B*T, C, imH, imW)
        x = (color_img - self.mean) / self.std

        x = self.encoder(x)

        if self.predict_depth:
            depth_logits = x[:, :self.D]
        else:
            depth_logits = depth_img.view(B*T, *depth_img.shape[2:])

        depth_context = x[:, self.D:(self.D + self.camC)]
        x = depth_logits.softmax(dim=1).unsqueeze(1) * depth_context.unsqueeze(2)

        geom = self.get_inv_geometry(intrinsics, extrinsics) 
        x = self.sample2bev(geom, x)

        x = x.view(B*T, -1, *x.shape[4:])
        pcloud = pcloud.view(B*T, *pcloud.shape[2:])
        debug = x

        if self.fuse_pcloud:
            x = torch.cat([x, pcloud], dim=1)

        x = self.bev_compressor(x)
        x = x.view(B, T, *x.shape[1:])
        x = self.temporal_model(x)
        x = x.view(B, -1, self.int_nx[0], self.int_nx[1])

        bev_features = self.decoder(x)
        bev_features = self.features_refine(bev_features)
        trav_map = self.travmap_head(bev_features)

        return trav_map, depth_logits, debug


