# temporal_fusion_STA.py
# 局部多头注意力机制+旋转编码+滑动窗口
# 参数为channels=64, temporal_length=6,  window_size=4, stride=2, num_heads=4

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RotaryPositionalEncoding(nn.Module):
    """旋转位置编码模块 (RoPE)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rope(self, x, seq_dim=-2):
        seq_len = x.size(seq_dim)
        device = x.device
        
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        emb = emb.view(*([1]*(x.ndim-2)), seq_len, self.dim)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return x * cos_emb + self._rotate_half(x) * sin_emb

class LocalMultiHeadAttention(nn.Module):
    """改进版局部多头注意力 (滑动窗口 + RoPE)"""
    def __init__(self, channels, window_size, num_heads, stride=2):
        super().__init__()
        assert channels % num_heads == 0, "channels必须能被num_heads整除"
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.stride = stride  # 新增滑动步长
        
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.rotary = RotaryPositionalEncoding(self.head_dim)
        
    def forward(self, x):
        B, C, T, H, W = x.shape if len(x.shape) == 5 else (0,0,0,0,0)
        orig_T = T if len(x.shape) == 5 else x.size(1)
        x = rearrange(x, 'b c t h w -> (b h w) t c') if len(x.shape) == 5 else x
        
        # 滑动窗口处理
        windows = []
        counts = torch.zeros((x.shape[0], T), device=x.device)  # 用于平均重叠区域
        
        # 生成滑动窗口
        for i in range(0, T - self.window_size + 1, self.stride):
            window = x[:, i:i + self.window_size]
            
            # QKV转换和注意力计算
            qkv = self.qkv(window).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads), qkv)
            
            # RoPE位置编码
            q = self.rotary.apply_rope(q, seq_dim=2)
            k = self.rotary.apply_rope(k, seq_dim=2)
            
            # 计算注意力
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=0.55, training=self.training)
            
            # 注意力输出
            out = attn @ v
            out = rearrange(out, 'b h l d -> b l (h d)')
            out = self.proj(out)
            
            # 记录窗口位置和计数
            windows.append((i, out))
            counts[:, i:i + self.window_size] += 1
            
        # 合并滑动窗口的输出
        output = torch.zeros((x.shape[0], T, C), device=x.device)
        for i, window_out in windows:
            output[:, i:i + self.window_size] += window_out
            
        # 处理重叠区域的平均值
        output = output / (counts.unsqueeze(-1) + 1e-8)
        
        # 还原维度
        if B != 0:
            output = rearrange(output, '(b h w) t c -> b c t h w', b=B, h=H, w=W)
        
        return output

class SpatioTemporalPooling(nn.Module):
    def __init__(self, in_channels, reduction_channels, pool_size):
        super().__init__()
        stride = (1, pool_size[1], pool_size[2])
        padding = (0, 0, 0)

        self.feature = nn.Sequential(
            nn.AvgPool3d(kernel_size=pool_size, stride=stride, padding=padding, ceil_mode=True),
            nn.Conv3d(in_channels, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(reduction_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x_pool = self.feature(x)
        _, _, t, h_target, w_target = x.size()
        x_pool = F.interpolate(
            x_pool, 
            size=(t, h_target, w_target),
            mode='trilinear',
            align_corners=False
        )
        return x_pool

class TemporalBlock(nn.Module):
    """改进TemporalBlock (适配RoPE)"""
    def __init__(self, channels, pool_size, window_size, num_heads):
        super().__init__()
        reduction_channels = channels // 4
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels//2),
            nn.GELU(),
            nn.Conv3d(channels//2, channels//2, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(channels//2),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels//2),
            nn.GELU()
        )
        
        self.pool = SpatioTemporalPooling(channels, reduction_channels, pool_size)
        self.aggregate = nn.Sequential(
            nn.Conv3d(channels + reduction_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU()
        )
        self.attn = LocalMultiHeadAttention(channels, window_size, num_heads)
        self.norm = nn.LayerNorm([channels])
        
    def forward(self, x):
        residual = x
        
        x1 = self.conv1(x)                                  # 提取空间特征
        x2 = self.conv2(x)                                  # 提取通道特征
        x_conv = torch.cat([x1, x2], dim=1)                 # 合并空间和通道特征
        
        x_pool = self.pool(x)                               # 时空池化，上下文融合，同时提取了时间和空间信息
        x_fused = torch.cat([x_conv, x_pool], dim=1)        # 合并特征，包括空间、通道和时间信息
        x_fused = self.aggregate(x_fused)                   # 特征更深度地融合，并且压缩维度，提取关键信息，减少计算
        
        x = residual + x_fused                              # 连接原始特征信息，防止信息丢失，解决梯度消失问题
        
        B, C, T, H, W = x.shape
        x_attn = self.attn(x)                               # 增强时序理解，捕捉时间依赖     
        x = self.norm(x_attn.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x

class TemporalModel(nn.Module):
    def __init__(self, channels, temporal_length, input_shape, window_size, num_heads):
        """
        时空池化模型，固定包含5个TemporalBlock模块
        
        Args:
            channels (int): 输入通道数
            temporal_length (int): 输入序列的时间步数
            input_shape (tuple): 输入张量的空间形状(高度，宽度)
            window_size (int): 注意力窗口的大小
            num_heads (int): 注意力头数量
        """
        super().__init__()
        self.temporal_length = temporal_length
        h, w = input_shape
        modules = []
        pool_size = (1, 2, 2)  # 保持时间维度完整的空间池化
        
        # 固定构建5个时空块
        for _ in range(5):
            temporal = TemporalBlock(
                channels=channels,
                pool_size=pool_size,
                window_size=window_size,
                num_heads=num_heads
            )
            modules.append(temporal)
            
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        # 输入形状验证
        assert x.size(1) == self.temporal_length, f"输入时间步数应为{self.temporal_length}，实际收到{x.size(1)}"
        
        # 维度调整: (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # 通过所有时空块
        x = self.model(x)
        
        # 恢复维度并返回最后一个时间步
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, -1, None]  # (B, 1, C, H, W)

# 测试代码
if __name__ == "__main__":
    # 参数配置
    channels = 64
    temporal_length = 6    # 可以自由设置时间步数
    input_shape = (100, 100)
    window_size = 4
    num_heads = 8

    # 实例化模型
    model = TemporalModel(
        channels=channels,
        temporal_length=temporal_length,
        input_shape=input_shape,
        window_size=window_size,
        num_heads=num_heads
    )
    
    # 验证模型结构
    print(f"模型包含的时空块数量: {len(model.model)}")  # 应该输出5
    
    # 测试前向传播
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    B = 2
    T = temporal_length
    C = channels
    H, W = input_shape
    input_tensor = torch.randn(B, T, C, H, W).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == (B, 1, C, H, W), "输出形状验证失败"
    print("测试通过！")
