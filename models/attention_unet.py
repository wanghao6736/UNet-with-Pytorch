"""
Attention UNet implementation using basic modules.
Based on the paper: https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .modules.attention import CBAM
from .modules.conv import DoubleConv


class AttentionUNet(nn.Module):
    """
    Attention UNet architecture for image segmentation.
    Adds attention gates to the standard UNet architecture.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of feature dimensions for each level
        bilinear (bool): Whether to use bilinear upsampling
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
        bilinear=True
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path
        for feature in features:
            self.downs.append(
                DoubleConv(
                    in_channels, feature,
                    use_batchnorm=True,
                    dropout_p=0.1
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(
            features[-1], features[-1] * 2,
            use_batchnorm=True,
            dropout_p=0.2
        )

        # Attention gates and up path
        for feature in reversed(features):
            # Attention gate
            self.attentions.append(CBAM(feature))

            # Upsampling block
            if bilinear:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(feature * 2, feature, kernel_size=1)
                    )
                )
            else:
                self.ups.append(
                    nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
                )

            # Double convolution after concatenation
            self.ups.append(
                DoubleConv(
                    feature * 2, feature,
                    use_batchnorm=True,
                    dropout_p=0.1
                )
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with attention
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # Resize if shapes don't match
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:], antialias=None)

            # Apply attention on skip connection
            skip = self.attentions[idx // 2](skip)

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = AttentionUNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
