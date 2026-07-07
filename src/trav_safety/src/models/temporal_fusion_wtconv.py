import torch
import torch.nn as nn
import torch.nn.functional as F

from .wtconv.wtconv2d import WTConv2d

class SpatioTemporalSeparableBlock(nn.Module):
    def __init__(self, channels, time_kernel=2, space_kernel=3, wt_levels=2):
        super().__init__()
        self.channel_compress = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 局部时间卷积
        self.temporal_local = nn.Sequential(
            nn.ConstantPad3d(padding=(0, 0, 0, 0, time_kernel-1, 0), value=0),
            nn.Conv3d(channels//2, channels//2, kernel_size=(time_kernel, 1, 1), 
                     groups=channels//2, bias=False)
        )
        
        # 全局时间上下文
        self.temporal_global = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),  # 时间维度全局池化
            nn.Conv3d(channels//2, channels//2, kernel_size=1),
            nn.BatchNorm3d(channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 时间特征融合gate
        self.temporal_gate = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间WTConv+点卷积处理，参考MobileNetV2的深度可分离卷积
        self.wtconv = WTConv2d(channels//2, channels//2, kernel_size=space_kernel, 
                             wt_levels=wt_levels, wt_type='db1')
        self.wtconv_bn = nn.BatchNorm2d(channels//2)  # WTConv的BN
        self.wtconv_relu = nn.ReLU(inplace=True)      # WTConv的ReLU
        
        self.pointwise = nn.Conv2d(channels//2, channels//2, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels//2)  # 点卷积的BN
        
        # 通道混合与归一化
        self.channel_mixing = nn.Sequential(
            nn.Conv3d(channels//2, channels//2, kernel_size=1),
            nn.BatchNorm3d(channels//2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 通道压缩
        x = self.channel_compress(x)
        
        # 局部时间卷积
        x_local = self.temporal_local(x)
        
        # 全局时间上下文
        x_global = self.temporal_global(x)
        x_global = x_global.expand_as(x_local)  # 扩展到与局部特征相同的尺寸
        
        # 融合局部和全局时间特征
        gate = self.temporal_gate(torch.cat([x_local, x_global], dim=1))
        x_temp = x_local * gate + x_global * (1-gate)
        x_temp = F.relu(x_temp)
        
        # 空间处理(WTConv+点卷积)
        B, C, T, H, W = x_temp.shape
        x_spatial = x_temp.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        
        # 深度卷积 → BN → ReLU
        x_spatial = self.wtconv(x_spatial)
        x_spatial = self.wtconv_bn(x_spatial)
        x_spatial = self.wtconv_relu(x_spatial)
        
        # 点卷积 → BN
        x_spatial = self.pointwise(x_spatial)
        x_spatial = self.pointwise_bn(x_spatial)
        
        # 恢复维度并混合通道
        x_out = x_spatial.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        x_out = self.channel_mixing(x_out)
        
        return x_out
        

class SpatioTemporalPooling(nn.Module):
    def __init__(self, in_channels, reduction_channels, pool_size):
        """
        时空池化层，作用是提取并保留时空维度上更具代表性的特征，减少噪声和冗余信息。

        Args:
            in_channels (int): 输入通道数
            reduction_channels (int): 通道减少后的输出通道数
            pool_size (tuple): 池化核的大小，包含三个元素，分别表示时间、高度和宽度
        
        Return:
            torch.Tensor: 时空池化后的张量
        """

        super().__init__()

        self.features = []
        stride = (1, *pool_size[1:])
        padding = (pool_size[0]-1, 0, 0)
        self.feature = nn.Sequential(
            torch.nn.AvgPool3d(kernel_size=pool_size, stride=stride, padding=padding, count_include_pad=False),
            nn.Conv3d(in_channels, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(reduction_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, _, t, h, w = x.shape
        x_pool = self.feature(x)[:, :, :-1].contiguous()
        c = x_pool.shape[1]
        x_pool = F.interpolate(x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w), mode='bilinear', align_corners=False)
        x_pool = x_pool.view(b, c, t, h, w)

        return x_pool

class TemporalBlock(nn.Module):
    def __init__(self, channels, pool_size):
        """
        时空池化模块，用时空池化层为基础

        Args:
            channels (int): 输入通道数
            pool_size (tuple): 池化核的大小，包含三个元素，分别表示时间、高度和宽度
        """
        super().__init__()

        self.conv1 = SpatioTemporalSeparableBlock(
            channels=channels,
            time_kernel=2,
            space_kernel=3,
            wt_levels=4
        )
        
        self.conv2 = nn.Sequential(
                nn.Conv3d(channels, channels // 2, kernel_size=1, bias=False),
                nn.BatchNorm3d(channels // 2),
                nn.ReLU(inplace=True))
        
        reduction_channels = channels // 3

        self.pyramid_pooling = SpatioTemporalPooling(channels, reduction_channels, pool_size)
        
        agg_channels = 2 * (channels // 2) + reduction_channels

        self.aggregation = nn.Sequential(
                    nn.Conv3d(agg_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_residual = torch.cat([x1, x2], dim=1)
        x_pool = self.pyramid_pooling(x)
        x_residual = torch.cat([x_residual, x_pool], dim=1)
        x_residual = self.aggregation(x_residual)

        x = x + x_residual
        
        return x

class TemporalModel(nn.Module):
    def __init__(self, channels, temporal_length, input_shape):
        """
        时空池化模型，用时空池化模块为基础

        Args:
            channels (int): 输入通道数
            temporal_length (int): 时间维度的长度
            input_shape (tuple): 输入张量的形状(高度，宽度)
        """
        super().__init__()

        h, w = input_shape
        modules = []
        for _ in range(temporal_length - 1):
            temporal = TemporalBlock(channels, pool_size=(2, h, w))
            modules.extend(nn.Sequential(temporal))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, -1, None]
    
if __name__ == "__main__":
    # 测试 TemporalModel 模块

    # 定义模型参数
    channels = 64          # 设置为64，确保channels能被num_heads整除
    temporal_length = 6    # 使用最近6帧
    input_shape = (100, 100) # 输入图像的高度和宽度

    model = TemporalModel(
        channels=channels,
        temporal_length=temporal_length,
        input_shape=input_shape
    )
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置模型为评估模式
    model.eval()

    # 创建模拟输入数据
    B = 2                  # 批量大小
    T = temporal_length    # 时间步数
    C = channels           # 通道数
    H, W = input_shape     # 高度和宽度

    # 创建随机输入张量
    input_tensor = torch.randn(B, T, C, H, W).to(device)  # (B, T, C, H, W)

    # 执行前向传播
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输入和输出的形状
    print(f"Input Shape: {input_tensor.shape}")   # (B, T, C, H, W)
    print(f"Output Shape: {output.shape}")        # (B, 1, C, H, W)

    # 验证输出形状是否正确
    assert output.shape == (B, 1, C, H, W), "输出形状不匹配预期值。"

    print("TemporalModel 前向传播测试通过！")