import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_fusion import TemporalModel
from .encoder_decoder import Encoder, Decoder

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

        # 模型参数初始化
        self.grid_bounds    = grid_bounds
        self.input_size     = (input_size[1], input_size[0]) # 从(H, W)换到(W, H)
        self.camC           = image_dim
        self.downsample     = downsample
        self.predict_depth  = predict_depth
        self.fuse_pcloud    = fuse_pcloud
        self.eps            = 1e-6
        

        '''
            _C.MODEL.GRID_BOUNDS = {             # 体素网格各个维度上的边界
                 'xbound': [-2.0, 8.0, 0.1],     # x 轴边界为 [-2.0, 8.0]，步长为 0.1
                 'ybound': [-5.0, 5.0, 0.1],     # y 轴边界为 [-5.0, 5.0]，步长为 0.1
                 'zbound': [-1.0, 2.0, 0.1],     # z 轴边界为 [-2.0, 2.0]，步长为 0.1
                 'dbound': [ 0.3, 8.0, 0.2]      # 深度轴 d 边界为 [0.3, 8.0]，步长为 0.2
            }
        '''

        # 遍历grid_bounds字典，分别获取x、y、z方向的步长，为[0.1, 0.1, 0.1]
        dx = torch.Tensor([row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        # 遍历grid_bounds字典，分别获取x、y、z方向的第一个体素的中心坐标，为[-1.95, -4.95, -1.95]
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        # 遍历grid_bounds字典，分别获取x、y、z方向的体素数量，为[100， 100， 40]
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        
        # 将nx移动到CPU上并从张量中分离出来，转换为numpy数组
        self.int_nx = nx.cpu().detach().numpy()

        # 将dx、bx、nx全部设置为不可训练的超参数
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # 创建图像标准化均值张量，维度(b,c,h,w)分别为(1,3,1,1)，通常用于预训练的模型(ImageNet上)
        mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        # 创建图像标准化标准差张量，维度(b,c,h,w)分别为(1,3,1,1)，通常用于预训练的模型(ImageNet上)
        std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()
        # 将均值和标准差设置为不可训练的超参数
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

        # 创建一个3D体素网格空间
        self.voxels = self.create_voxels()

        # 如果预测深度
        if self.predict_depth:
            # 则计算深度轴上的坐标数量，self.D = 38
            self.D = int((grid_bounds['dbound'][1] - grid_bounds['dbound'][0]) / grid_bounds['dbound'][2])
        else:
            # 否则设置深度坐标数量为0
            self.D = 0

        # 定义潜在维度，64
        self.latent_dim = image_dim

        # 定义点云融合维度，若fuse_pcloud为True，则计算点云维度，否则设置为0
        if self.fuse_pcloud:
            # 点云维度为z轴上的体素数量
            pcloud_dim = self.nx[2]
        else:
            pcloud_dim = 0
        
        # 实例化encoder(2D的)，输出通道数为 D + camC ，用于提取BEV特征
        self.encoder = Encoder(self.D + self.camC, downsample=self.downsample)

        # BEV压缩器，是一个神经网络块，需要训练，用于将转换为BEV视角的特征继续压缩提取，提高效率        
        self.bev_compressor = nn.Sequential(
            # 二维卷积层，输入通道数为camC * nx[2] + pcloud_dim，输出通道数为camC，卷积核大小为3x3，填充为1
            nn.Conv2d(self.camC * self.nx[2] + pcloud_dim, self.camC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.camC),
            nn.ReLU(inplace=True))

        self.temporal_model = TemporalModel(
            channels=self.camC,
            temporal_length=temporal_length,
            input_shape=(nx[0], nx[1]))

        # 实例化decoder(2D的)
        self.decoder = Decoder(in_channels=self.latent_dim)
        
        # 可通行性代价地图头，用于生成可通行性代价地图
        self.travmap_head = nn.Sequential(
            # 二维卷积层，输入输出的通道数和尺寸均不变
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
            # 二维卷积层，通道数从latent_dim变为2，分别表示预测的mu和nu，输出尺寸不变
            nn.Conv2d(self.latent_dim, 2, kernel_size=1, padding=0),
            # 二维Sigmoid激活函数，将输出的概率值映射到[0, 1]之间
            nn.Sigmoid())
    
    def create_voxels(self):
        """
        创建一个3D体素网格空间，这个空间在世界坐标系下，原点(0, 0, 0)就是机器人的投影点

        Returns:
            torch.nn.Parameter: 3D体素网格

        _C.MODEL.GRID_BOUNDS = {                 # 体素网格各个维度上的边界
                 'xbound': [-2.0, 8.0, 0.1],     # x 轴边界为 [-2.0, 8.0]，步长为 0.1
                 'ybound': [-5.0, 5.0, 0.1],     # y 轴边界为 [-5.0, 5.0]，步长为 0.1
                 'zbound': [-1.0, 2.0, 0.1],     # z 轴边界为 [-2.0, 2.0]，步长为 0.1
                 'dbound': [ 0.3, 8.0, 0.2]      # 深度轴 d 边界为 [0.3, 8.0]，步长为 0.2
            }
        """
        # 创建z轴上的体素网格
        # 生成z轴上的体素坐标，[1.9, 1.8, ..., -1.9, -2.0]，共40个坐标
        grid_z = torch.arange(*self.grid_bounds['zbound'], dtype=torch.float).flip(0)
        # 将z轴上的体素坐标从1维调整到3维，形状为[40, 1, 1]
        # 可以想象为现在只是3维空间上的一个轴，这个轴在z方向上，有40个点
        grid_z = torch.reshape(grid_z, [self.nx[2], 1, 1])
        # 在 x 和 y 轴上进行重复，最终的 grid_z 张量形状为 [40, 100, 100]
        # 继续想象，先是让这个轴在x轴上复制100次，然后用这100个轴再在y轴上复制100次
        grid_z = grid_z.repeat(1, self.nx[0], self.nx[1])

        # 创建y轴上的体素网格
        # [4.9, 4.8, 4.7, ..., -4.9, -5.0]，共100个坐标
        grid_y = torch.arange(*self.grid_bounds['ybound'], dtype=torch.float).flip(0)
        # 形状变为[1, 1, 100],同理现在为y轴方向上的一个轴，在y轴方向上，有100个点
        grid_y = torch.reshape(grid_y, [1, 1, self.nx[1]])
        # 形状变为[40, 100, 100]，同理，将这个轴在x轴上复制100次，然后用这100个轴再在z轴上复制40次
        grid_y = grid_y.repeat(self.nx[2], self.nx[0], 1)

        # 创建x轴上的体素网格
        # [7.9, 7.8, 7.7, ..., -1.9, -2.0]，共100个坐标
        grid_x = torch.arange(*self.grid_bounds['xbound'], dtype=torch.float).flip(0)
        # 形状变为[1, 100, 1]，同理现在为x轴方向上的一个轴，在x轴方向上，有100个点
        grid_x = torch.reshape(grid_x, [1, self.nx[0], 1])
        # 形状变为[40, 100, 100]，同理，将这个轴在y轴上复制100次，然后用这100个轴再在z轴上复制40次
        grid_x = grid_x.repeat(self.nx[2], 1, self.nx[1])

        # Z x X x Y x 3
        # 将grid_x、grid_y、grid_z沿第四维度拼接，得到3D体素网格，表示每个体素在3D空间中的坐标
        # 例如：voxels[0, 0, 0] = [7.9, 4.9, 1.9]，表示第一个体素在3D空间中的坐标为(7.9, 4.9, 1.9)
        voxels = torch.stack((grid_x, grid_y, grid_z), -1)
        return nn.Parameter(voxels, requires_grad=False)
    
    def get_inv_geometry(self, intrinsics, extrinsics):
        """
        转换体素网格，从世界坐标系 -> 相机坐标系 -> 图像平面坐标系

        Args:
            intrinsics (torch.Tensor): 相机内参矩阵，(B, 3, 3)
            extrinsics (torch.Tensor): 相机外参矩阵，包括旋转矩阵和平移向量，(B, 4, 4)

        Returns:
            torch.Tensor: 图像平面坐标系中的体素网格（体素坐标）
        """
        # 从外参矩阵中提取旋转矩阵和平移向量
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        # 获取batch大小、相机数量
        B, N, _ = translation.shape
        # 对体素网格张量进行扩展，在前面添加batch维度B、相机数量维度N，在后面添加一个虚拟维度，最后的体素张量形状为[B, 1, 40, 100, 100, 3, 1]
        points = self.voxels.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = points.expand(B, -1, -1, -1, -1, -1, -1)
        # 将体素网格张量从世界坐标系转换到相机坐标系
        # 先平移
        points = points - translation.view(B, N, 1, 1, 1, 3, 1)
        # 将内参矩阵与旋转矩阵的逆矩阵相乘，得到组合变换矩阵combined_transformation
        combined_transformation = intrinsics.matmul(torch.inverse(rotation))
        # 再组合变换，用combined_transformation将体素网格张量转换到相机坐标系，并移除虚拟维度，形状为[B, N, 40, 100, 100, 3]
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        """
        透视投影（Perspective Projection） 是一种将三维空间中的点映射到二维平面（图像平面）上的方法，它模拟了人眼或相机的成像过程。
        在透视投影中，随着物体距离观察者越远（深度(z)越大），它在投影平面上的呈像就越小，这符合现实中看到的视觉效果。
        其中，x和y坐标除以深度z这个操作是透视投影的核心，实现距离越远的点在投影平面上越接近中心，模拟真实的视觉效果。
        而拼接原z坐标是因为在其他任务中，仍然需要知道点的深度信息，所以保留了z坐标。
        """

        # points[..., :2]取出体素的x，y坐标，points[..., 2:3]取出体素的z坐标，将体素的x，y坐标除以z坐标，得到归一化的x 和 y 坐标
        # torch.cat将归一化的x，y坐标和z坐标拼接在一起，得到透视投影到图像平面坐标系的体素网格
        points = torch.cat((points[..., :2] / (points[..., 2:3] + self.eps), points[..., 2:3]), -1)

        return points
    
    def sample2bev(self, geometry, x):
        """
        用图像平面坐标系的体素坐标对输入特征图进行采样，生成BEV特征，坐标系转换到BEV空间
        
        Args:
            geometry (torch.Tensor): 
                转换到相机平面坐标系的体素坐标张量，形状为 [batch, T, Z, X, Y, 3]。
                其中：
                    - batch: 一共几个batch，为4
                    - T: 时间步数，即一个批次有几帧图，为6
                    - Z, X, Y: 体素地图的深度、高度和宽度
                    - 3: 坐标 (u, v, depth)
            
            x (torch.Tensor): 
                输入的特征图张量，形状为 [batch, D, H, W, C]。
                其中：
                    - batch: 一共几个batch，为4
                    - D: 特征图的深度（通常对应于通道数或其他维度）
                    - H, W: 特征图的高度和宽度
                    - C: 特征图的通道数
        
        Returns:
            torch.Tensor: 
                采样后的BEV特征张量，形状为 [batch, T, D, H_out, W_out, C]。
                其中：
                    - H_out, W_out: 经过采样后的BEV特征图的高度和宽度
        """
    
        # 获取输入特征图的形状，d为深度，h为高度，w为宽度，c为通道数
        _, d, h, w, c = x.shape
        # 获取体素坐标的形状，batch为批量大小，T为相机数量，Z, X, Y为体素地图的深度、高度和宽度
        batch, T, Z, X, Y, _ = geometry.shape
        # 将输入体素的形状从 [batch, T, Z, X, Y, 3] 调整为 [batch*T, Z, X, Y, 3]
        geometry = geometry.view(batch*T, Z, X, Y, 3)

        # geometry[..., 0]提取geometry张量中所有点的x坐标，归一化后映射到[-1, 1]之间，记为u坐标
        u = 2 * geometry[..., 0] / (self.input_size[1]-1) - 1
        # geometry[..., 1]提取geometry张量中所有点的y坐标，归一化后映射到[-1, 1]之间，记为v坐标
        v = 2 * geometry[..., 1] / (self.input_size[0]-1) - 1
        # geometry[..., 2]提取geometry张量中所有点的深度坐标，归一化后映射到[-1, 1]之间，记为depth坐标
        depth = 2 * (geometry[...,2] - self.grid_bounds['dbound'][0]) / (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0]) - 1
        
        # 将u、v、depth张量沿第四维度堆叠，得到采样网格grid，表示相对于输入特征图的空间位置，形状为 [batch*T, Z, X, Y, 3]
        grid = torch.stack((u, v, depth), -1)

        # 通过grid_sample函数对输入特征图进行采样，得到采样后的BEV特征
        x = F.grid_sample(x, grid, align_corners=False)
        
        # 将采样后的BEV特征的形状从 [batch*T, D, H, W, C] 调整为 [batch, T, D, H, W, C]
        x = x.view(batch, T, *x.shape[1:])
        return x
        
    def forward(self, color_img, pcloud, intrinsics, extrinsics, depth_img=None):
        """
        TravNet模型的前向传播

        Args:
            color_img (torch.Tensor): 彩色图像张量，形状为 (B, T, C, H, W).
            pcloud (torch.Tensor): 点云张量，形状为 (B, T, Z, Y, X).
            intrinsics (torch.Tensor): 内参矩阵张量，形状为 (B, T, 3, 3).
            extrinsics (torch.Tensor): 外参矩阵张量，形状为 (B, T, 4, 4).
            depth_img (torch.Tensor, optional): 深度图像张量，可选，默认为None.

        Returns:
            tuple: A tuple containing:
                - trav_map (torch.Tensor): 可通过性地图，形状为 (B, 2, X, Y).
                - depth_logits (torch.Tensor): 预测的原始深度值，形状为 (B*T, D, H, W).
                - debug (torch.Tensor): debug调试信息
        """

        # 确保输入彩色图像的通道数为3
        B, T, C, imH, imW = color_img.shape
        assert(C==3)

        # 将输入彩色图像重塑为 (B*T, C, imH, imW)的形状
        color_img = color_img.view(B*T, C, imH, imW)

        # 标准化彩色图像
        x = (color_img - self.mean) / self.std

        # 传入encoder进行特征提取
        x = self.encoder(x)

        # Depth is B x N x D x H/downsample x W/downsample
        # 如果需要预测深度，则从encoder的输出中提取深度信息，否则直接使用输入的深度图像
        if self.predict_depth:
            # 提取特征的前38个通道，作为深度预测值
            depth_logits = x[:, :self.D]
        else:
            depth_logits = depth_img.view(B*T, *depth_img.shape[2:])

        # 从encoder的输出中提取深度上下文信息，并将其与深度预测值相乘，进行融合
        depth_context = x[:, self.D:(self.D + self.camC)]
        # depth_logits 形状为 B*N x D x H/downsample x W/downsample
        # depth_context 形状为 B*N x C x H/downsample x W/downsample
        x = depth_logits.softmax(dim=1).unsqueeze(1) * depth_context.unsqueeze(2)

        # 计算从体素地图到相机坐标系的逆几何变换矩阵，即相机坐标系到体素地图的变换矩阵
        # 计算图像平面坐标系中的体素网格
        geom = self.get_inv_geometry(intrinsics, extrinsics)
        # 将视锥中的采样点变换为BEV表示，x形状为[batch, T, D, H, W, C] 
        x = self.sample2bev(geom, x)

        # 重塑x的形状，从B x T x C x Z x X x Y重塑为B*T x C*Z x X x Y
        x = x.view(B*T, -1, *x.shape[4:])
        # 重塑点云的形状，从B x T x Z x X x Y重塑为B*T x Z x X x Y
        pcloud = pcloud.view(B*T, *pcloud.shape[2:])
        debug = x

        # 如果需要融合点云，则将点云与BEV特征进行拼接
        if self.fuse_pcloud:
            # Concatenate with pointcloud
            x = torch.cat([x, pcloud], dim=1)

        # 压缩BEV特征，至B x T x C x X x Y
        x = self.bev_compressor(x)

        # 恢复时间序列维度：B x T x C x X x Y
        x = x.view(B, T, *x.shape[1:])

        # 用一个3D卷积层temporal_model对BEV特征进行时间融合
        x = self.temporal_model(x)
        
        # 重塑x的形状为B x C x X x Y
        x = x.view(B, -1, self.int_nx[0], self.int_nx[1])
        
        # 用decoder对融合了时序的BEV特征进行解码，得到BEV特征
        # Decoder用于从经过了时序融合后的BEV特征中生成更高分辨率的BEV特征，用于计算可通行性代价地图。
        bev_features = self.decoder(x)
        
        # 计算可通行性代价地图  
        trav_map = self.travmap_head(bev_features)

        return trav_map, depth_logits, debug
