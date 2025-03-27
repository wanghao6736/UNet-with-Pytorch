"""
Basic convolution modules for UNet variants.
Contains implementations of different convolution blocks used across UNet architectures.
"""

from typing import Tuple, Type, Union

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolution block with configurable components.
    (convolution => [BN] => [activation] => [dropout])

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolution kernel
        padding (Union[int, Tuple[int, int]], optional): Padding size. Defaults to None
        use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True
        activation (Type[nn.Module], optional): Activation function to use. Defaults to nn.ReLU
        dropout_p (float, optional): Dropout probability. If 0, no dropout is applied. Defaults to 0
        order (tuple, optional): Order of operations. Defaults to ('conv', 'bn', 'act', 'dropout')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = None,
        use_batchnorm: bool = True,
        activation: Type[nn.Module] = nn.ReLU,
        dropout_p: float = 0,
        order: tuple = ('conv', 'bn', 'act', 'dropout')
    ):
        super().__init__()

        if padding is None:
            # Calculate same padding
            padding = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else \
                tuple((k - 1) // 2 for k in kernel_size)

        # Dictionary to map operation names to their implementations
        self.ops_dict = {
            'conv': lambda: nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batchnorm  # Disable bias when using BatchNorm
            ),
            'bn': lambda: nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            'act': lambda: activation(inplace=True) if activation else nn.Identity(),
            'dropout': lambda: nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        }

        # Build the sequential model based on the specified order
        layers = []
        for op in order:
            if op in self.ops_dict:
                layers.append(self.ops_dict[op]())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """
    Double convolution block used in UNet architectures.
    (convolution => [BN] => ReLU => [Dropout]) * 2

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (Union[int, Tuple[int, int]], optional): Kernel size. Defaults to 3
        use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True
        mid_channels (int, optional): Number of middle channels. Defaults to out_channels
        dropout_p (float, optional): Dropout probability. If 0, no dropout is applied. Defaults to 0
        deep_supervision (bool, optional): Whether to use deep supervision. Defaults to False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_batchnorm: bool = True,
        mid_channels: int = None,
        dropout_p: float = 0,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            ConvBlock(
                in_channels, mid_channels,
                kernel_size=kernel_size,
                use_batchnorm=use_batchnorm,
                dropout_p=dropout_p
            ),
            ConvBlock(
                mid_channels, out_channels,
                kernel_size=kernel_size,
                use_batchnorm=use_batchnorm,
                dropout_p=dropout_p
            )
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualDoubleConv(nn.Module):
    """
    Residual double convolution block.
    Adds a residual connection around the double convolution.
    Following ResNet's pre-activation design.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of middle channels. Defaults to out_channels
        dropout_p (float, optional): Dropout probability. If 0, no dropout is applied. Defaults to 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        dropout_p: float = 0
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = DoubleConv(
            in_channels, out_channels,
            mid_channels=mid_channels,
            dropout_p=dropout_p
        )

        if in_channels != out_channels:
            # 修复：在预激活设计中，BatchNorm 应该使用输入通道数
            self.residual = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.double_conv(x) + self.residual(x)


class FeatureFusion(nn.Module):
    """
    Feature fusion module for NestedUNet.
    Fuses features from multiple scales.

    Args:
        in_channels_list (list): List of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBlock(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

    def forward(self, x_list):
        out = sum([conv(x) for conv, x in zip(self.convs, x_list)])
        return out
