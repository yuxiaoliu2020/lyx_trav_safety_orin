import torch
import torch.nn as nn
import torch.nn.functional as F
    
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

        # 先调用父类(nn.Module)的构造函数
        super().__init__()

        # 初始化一个空的时空池化层
        self.features = []

        # 定义池化步长，是一个元组，第一个元素为1，后面的元素为pool_size的第二个元素到最后一个元素
        # 这样定义是为了让池化操作在时间维度上不进行下采样，而只在高度和宽度维度上进行下采样，从而减少特征图的空间尺寸的同时保留时间维度的信息。
        stride = (1, *pool_size[1:])
        # 定义padding，是一个元组，第一个元素为pool_size的第一个元素减1，后面的元素为0
        # 这样定义是为了让池化操作在时间维度上不进行padding，而只在高度和宽度维度上进行padding，从而确保时间维度的边界信息不丢失。
        padding = (pool_size[0]-1, 0, 0)
        # 定义一个池化、卷积、BatchNorm 和 ReLU 的序列，作为时空池化层
        self.feature = nn.Sequential(
            # 三维平均池化层，池化核大小为pool_size，步长为stride，padding为padding，不在计算平均值时包含padding的值
            torch.nn.AvgPool3d(kernel_size=pool_size, stride=stride, padding=padding, count_include_pad=False),
            # 三维卷积层
            nn.Conv3d(in_channels, reduction_channels, kernel_size=1, bias=False),
            # 三维BatchNorm层
            nn.BatchNorm3d(reduction_channels),
            # 三维ReLU层
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
            )
        
    # 时空池化层的前向传播
    def forward(self, x):
        # 获取输入张量的维度，只取batch、时间、高度和宽度
        b, _, t, h, w = x.shape
        # 先用时空池化层对x进行池化和卷积，然后在时间维度上裁减，去掉最后一个时间步的特征，.contiguous()用于保证张量在内存中连续
        x_pool = self.feature(x)[:, :, :-1].contiguous()
        # 获取通道数
        c = x_pool.shape[1]
        # 先将x_pool的时间和批量维度合并，便于在空间维度上插值，最后将高度和宽度调整为原尺寸(h, w)
        x_pool = F.interpolate(x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w), mode='bilinear', align_corners=False)
        # 将x_pool恢复到(b, c, t, h, w)维
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

        # 定义conv1
        self.conv1 = nn.Sequential(
            # 三维卷积层，输入通道数为channels，输出通道数为channels/2，卷积核大小为1x1x1，说明不改变特征图的空间尺寸
            nn.Conv3d(channels, channels // 2, kernel_size=1, bias=False),
            # 三维BatchNorm层
            nn.BatchNorm3d(channels // 2),
            # 三维ReLU层
            nn.ReLU(inplace=True),
            # 三维常量填充层，顺序为(上填充(高度)，下填充(高度)，左填充(宽度)，右填充(宽度)，前填充(时间)，后填充(时间))
            # padding的最后一个值为0表明时间维度的后面不填充，value为0指用0填充
            nn.ConstantPad3d(padding=(1, 1, 1, 1, 1, 0), value=0),
            # 三维卷积层，输入通道数为channels/2，输出通道数为channels/2，卷积核大小为(2, 3, 3)
            nn.Conv3d(channels // 2, channels // 2, kernel_size=(2, 3, 3), bias=False),
            # 三维BatchNorm层
            nn.BatchNorm3d(channels // 2),
            # 三维ReLU层
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
        
        # 定义conv2
        self.conv2 = nn.Sequential(
                # 三维卷积层，输入通道数为channels，输出通道数为channels/2，卷积核大小为1x1x1，说明不改变特征图的空间尺寸
                nn.Conv3d(channels, channels // 2, kernel_size=1, bias=False),
                # 三维BatchNorm层
                nn.BatchNorm3d(channels // 2),
                # 三维ReLU层
                nn.ReLU(inplace=True),
                nn.Dropout3d(0.1)
                )
                
        
        # 金字塔池化层的输出通道数，为输入通道数的1/3
        reduction_channels = channels // 3
        # 定义金字塔池化层
        self.pyramid_pooling = SpatioTemporalPooling(channels, reduction_channels, pool_size)
        
        # 特征增强层的输出通道数，为输入通道数的2/3
        agg_channels = 2 * (channels // 2) + reduction_channels

        # 定义特征增强层，包括一个1x1x1卷积层、一个BatchNorm层和一个ReLU层
        self.aggregation = nn.Sequential(
                    nn.Conv3d(agg_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(channels),
                    nn.ReLU(inplace=True))

    # 时空池化模块的前向传播
    def forward(self, x):
        x1 = self.conv1(x)  # channels -> channels // 2；(T,H,W) -> (T,H,W)
        x2 = self.conv2(x)  # channels -> channels // 2；(T,H,W) -> (T,H,W)
        x_residual = torch.cat([x1, x2], dim=1)  # channels // 2 + channels // 2 -> channels；(T,H,W)
        x_pool = self.pyramid_pooling(x)         # channels -> channels // 3；(T,H,W) -> (T,H,W)
        x_residual = torch.cat([x_residual, x_pool], dim=1)  # channels + channels // 3；(T,H,W)
        x_residual = self.aggregation(x_residual)  # channels + channels // 3 -> channels；(T,H,W) -> (T,H,W)

        x = x + x_residual  # channels + channels -> channels；(T,H,W) -> (T,H,W)
        
        return x  # channels；(T,H,W)

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

        # 提取输入张量的高度和宽度
        h, w = input_shape
        # 定义一个空的模型
        modules = []
        # 遍历时间维度的长度
        for _ in range(temporal_length - 1):
            # 实例化一个时空池化模块，输入通道数为channels，池化核大小为(2, h, w)
            temporal = TemporalBlock(channels, pool_size=(2, h, w))
            # 创建temporal_length个TemporalBlock模块，组成一个顺序的神经网络层次结构
            modules.extend(nn.Sequential(temporal))

        # 将modules中的模块组合成一个顺序的神经网络model
        self.model = nn.Sequential(*modules)

    # 时空池化模型的前向传播
    def forward(self, x):
        # 将输入张量的维度从(b, t, c, h, w)调整为(b, c, t, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # 将x输入到模型中
        x = self.model(x)
        # 将x的维度从(b, c, t, h, w)调整回(b, t, c, h, w)，并保证张量在内存中连续
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # 时间维度的长度减少为1，即(b, c, 1, h, w)，并且时间维度上只保留x最后一个时间步的特征
        return x[:, -1, None]