"""
Attention modules for UNet variants.
Implements different attention mechanisms including channel attention,
spatial attention and combined attention (CBAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Computes attention weights for each channel using both max-pooling and avg-pooling features
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = channels // reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 提取共同参数配置
        shared_kwargs = {
            'kernel_size': 1,
            'bias': False
        }

        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, **shared_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, **shared_kwargs)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Computes attention weights for each spatial location using channel-wise statistics
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='reflect'  # 增加边缘处理
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines both channel and spatial attention
    """

    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
