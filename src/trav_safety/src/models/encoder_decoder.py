import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet34

class UpsamplingConcat(nn.Module):
    """上采样后拼接模块
    Note:
        拼接操作在通道维度上将两个特征图连接在一起，改变了特征图的通道数，但不会改变特征图的尺寸。
        拼接的主要作用是保留并融合其他来源的重要特征，使得后续的卷积操作能够同时处理这些特征，从而提高模型的表达能力和性能。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scale_factor (int, optional): 上采样因子，默认为2
    
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        # 定义上采样层，使用双线性插值
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # 定义卷积层，用于特征提取，包括2个2D卷积层、2个BatchNorm层和2个ReLU激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x_to_upsample, x):
        """上采样后拼接模块的前向传播

        Args:
            x_to_upsample (torch.Tensor): 需要上采样的张量
            x (torch.Tensor): 需要与上采样张量拼接的张量

        Returns:
            torch.Tensor: 上采样、拼接再卷积后的张量
        """
        # 将x_to_upsample上采样
        x_to_upsample = self.upsample(x_to_upsample)
        # 计算x_to_upsample和x的尺寸差异
        diffY = x.size()[2] - x_to_upsample.size()[2]
        diffX = x.size()[3] - x_to_upsample.size()[3]
        # 使用F.pad函数对x_to_upsample进行填充，使其与x尺寸匹配
        x_to_upsample = F.pad(x_to_upsample, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # 将x和x_to_upsample拼接，这种拼接是沿通道维度的
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    """上采样后相加模块
    
    Note:
        两个特征图相加指的是对应位置的像素值逐元素相加。
        这种方式可以将不同特征图的信息融合在一起，生成一个新的特征图。
        在ResNet和跳跃连接（skip connections）中，通过相加可以保留和融合不同层次的特征。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scale_factor (int, optional): 上采样因子，默认为2
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # 定义一个上采样layer，包括一个双线性插值上采样层、一个1x1卷积层和一个BatchNorm层
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x, x_skip):
        """上采样后相加模块的前向传播

        Args:
            x (torch.Tensor): 需要被上采样的张量
            x_skip (torch.Tensor): 需要与上采样张量相加的张量

        Returns:
            torch.Tensor: 上采样、相加后的张量
        """
        # 对x进行上采样，输出通道数为out_channels
        x = self.upsample_layer(x)
        # 计算x和x_skip的尺寸差异
        diffY = x_skip.size()[2] - x.size()[2]
        diffX = x_skip.size()[3] - x.size()[3]
        # 使用F.pad函数对x进行填充，使其与x_skip尺寸匹配
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # 返回x和x_skip相加的结果
        return x + x_skip

class Decoder(nn.Module):
    """
    解码器模块，用于特征提取和上采样

    Args:
        in_channels (int): 输入通道数
    """
    def __init__(self, in_channels):
        super().__init__()
        # 使用ResNet18作为骨干网络，不加载预训练权重，初始化残差块的权重
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        # first_conv用2d卷积层，输出通道数为64
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # bn1用backbone的bn1，relu用backbone的relu
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        # layer1、layer2、layer3也用backbone的layer1、layer2、layer3
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # up3_skip、up2_skip、up1_skip均用UpsamplingAdd模块
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, in_channels, scale_factor=2)

    def forward(self, x):
        """
        解码器的前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 解码并上采样后的张量
        """
        
        skip_1 = x                      # (H, W)
        x = self.first_conv(x)          # (H, W) -> (H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)        

        skip_2 = self.layer1(x)         # (H/2, W/2) -> (H/2, W/2)
        x = self.layer2(skip_2)         # (H/2, W/2) -> (H/4, W/4)

        skip_3 = x                      # (H/4, W/4)
        x = self.layer3(skip_3)         # (H/4, W/4) -> (H/8, W/8)

        x = self.up3_skip(x, skip_3)    # UpsamplingAdd是将x的尺寸适应到x_skip的尺寸，所以为(H/4, W/4)      
        x = self.up2_skip(x, skip_2)    # (H/2, W/2)
        x = self.up1_skip(x, skip_1)    # (H, W)

        return x

class Encoder(nn.Module):
    """
    编码器模块，用于提取BEV特征和上采样

    Args:
        C (int): 输出通道数
        downsample (int, optional): 下采样因子，默认为8

    Returns:
        torch.Tensor: 编码并上采样后的张量，形状为(H, W, C)
    """
    def __init__(self, C, downsample=8):
        super().__init__()
        self.C = C
        self.downsample = downsample

        print('Using Resnet34')
        resnet = resnet34(pretrained=True)

        c0 = 64
        c1 = 128
        c2 = 256

        # 当下采样因子为8时
        if downsample == 8:
            # backbone用resnet34的的初始卷积层、最大池化层、layer1和layer2
            self.backbone = nn.Sequential(*list(resnet.children())[:-4])
            # layer用resnet34的layer3
            self.layer = resnet.layer3
            # upsample_layer用UpsamplingConcat模块
            self.upsampling_layer = UpsamplingConcat(c2+c1, c1)
            # depth_layer用1x1卷积层，输出通道数为C
            self.depth_layer = nn.Conv2d(c1, self.C, kernel_size=1, padding=0)
        
        # 当下采样因子为4时
        elif downsample == 4:
            # backbone用resnet34的的初始卷积层、最大池化层和layer1
            self.backbone = nn.Sequential(*list(resnet.children())[:-5])
            # layer1用resnet34的layer2
            self.layer1 = resnet.layer2
            # layer2用resnet34的layer3
            self.layer2 = resnet.layer3
            # upsample_layer1、upsample_layer2用UpsamplingConcat模块
            self.upsampling_layer1 = UpsamplingConcat(c2+c1, c1)
            self.upsampling_layer2 = UpsamplingConcat(c1+c0, c0)
            # depth_layer用1x1卷积层，输出通道数为C
            self.depth_layer = nn.Conv2d(c0, self.C, kernel_size=1, padding=0)

        # 不支持其他下采样因子
        else:
            print('Downsample {} not implemented'.format(downsample))
            sys.exit(1)

    def forward(self, x):
        """
        编码器的前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 编码并上采样后的张量
        """
        x1 = self.backbone(x)                   # (H, W) -> (H/8, W/8)    
        if self.downsample == 8:
            x = self.layer(x1)                  # (H/8, W/8) -> (H/16, W/16)
            x = self.upsampling_layer(x, x1)    # 与x1的尺寸匹配，所以(H/16, W/16) -> (H/8, W/8)
        elif self.downsample == 4:
            x2 = self.layer1(x1)                # (H/8, W/8) -> (H/16, W/16)
            x = self.layer2(x2)                 # (H/16, W/16)
            x = self.upsampling_layer1(x, x2)   # 与x2的尺寸匹配，所以(H/16, W/16) -> (H/8, W/8)
            x = self.upsampling_layer2(x, x1)   # 与x1的尺寸匹配，所以(H/8, W/8) -> (H/4, W/4)
        x = self.depth_layer(x)                 # 最后尺寸不变，调整通道数为C

        return x