"""
Nested UNet (UNet++) implementation using basic modules.
Based on the paper: https://arxiv.org/abs/1807.10165
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .modules.conv import DoubleConv, FeatureFusion


class NestedUNet(nn.Module):
    """
    Nested UNet (UNet++) architecture with dense skip connections
    and optional deep supervision.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of feature dimensions for each level
        deep_supervision (bool): Whether to use deep supervision
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[32, 64, 128, 256, 512],
        deep_supervision=False
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Common operations
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # First column (encoder path)
        self.conv0_0 = DoubleConv(in_channels, features[0])
        self.conv1_0 = DoubleConv(features[0], features[1])
        self.conv2_0 = DoubleConv(features[1], features[2])
        self.conv3_0 = DoubleConv(features[2], features[3])
        self.conv4_0 = DoubleConv(features[3], features[4])

        # Nested dense skip pathways
        # First level
        self.conv0_1 = DoubleConv(features[0] + features[1], features[0])
        self.conv1_1 = DoubleConv(features[1] + features[2], features[1])
        self.conv2_1 = DoubleConv(features[2] + features[3], features[2])
        self.conv3_1 = DoubleConv(features[3] + features[4], features[3])

        # Feature fusion modules for combining multiple inputs
        self.fusion0_2 = FeatureFusion([features[0]] * 2 + [features[1]], features[0])
        self.fusion1_2 = FeatureFusion([features[1]] * 2 + [features[2]], features[1])
        self.fusion2_2 = FeatureFusion([features[2]] * 2 + [features[3]], features[2])

        self.fusion0_3 = FeatureFusion([features[0]] * 3 + [features[1]], features[0])
        self.fusion1_3 = FeatureFusion([features[1]] * 3 + [features[2]], features[1])

        self.fusion0_4 = FeatureFusion([features[0]] * 4 + [features[1]], features[0])

        # Final layers for deep supervision
        if deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channels, 1)
            self.final2 = nn.Conv2d(features[0], out_channels, 1)
            self.final3 = nn.Conv2d(features[0], out_channels, 1)
            self.final4 = nn.Conv2d(features[0], out_channels, 1)
        else:
            self.final = nn.Conv2d(features[0], out_channels, 1)

    def _size_check(self, up_feat, skip_feat):
        """Resize feature maps if needed"""
        if up_feat.shape[2:] != skip_feat.shape[2:]:
            up_feat = TF.resize(up_feat, size=skip_feat.shape[2:], antialias=None)
        return up_feat

    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested path 1
        x0_1 = self.conv0_1(torch.cat([
            x0_0, self._size_check(self.up(x1_0), x0_0)
        ], 1))

        x1_1 = self.conv1_1(torch.cat([
            x1_0, self._size_check(self.up(x2_0), x1_0)
        ], 1))

        x2_1 = self.conv2_1(torch.cat([
            x2_0, self._size_check(self.up(x3_0), x2_0)
        ], 1))

        x3_1 = self.conv3_1(torch.cat([
            x3_0, self._size_check(self.up(x4_0), x3_0)
        ], 1))

        # Nested path 2
        x0_2 = self.fusion0_2([
            x0_0, x0_1, self._size_check(self.up(x1_1), x0_0)
        ])

        x1_2 = self.fusion1_2([
            x1_0, x1_1, self._size_check(self.up(x2_1), x1_0)
        ])

        x2_2 = self.fusion2_2([
            x2_0, x2_1, self._size_check(self.up(x3_1), x2_0)
        ])

        # Nested path 3
        x0_3 = self.fusion0_3([
            x0_0, x0_1, x0_2, self._size_check(self.up(x1_2), x0_0)
        ])

        x1_3 = self.fusion1_3([
            x1_0, x1_1, x1_2, self._size_check(self.up(x2_2), x1_0)
        ])

        # Final nested path
        x0_4 = self.fusion0_4([
            x0_0, x0_1, x0_2, x0_3, self._size_check(self.up(x1_3), x0_0)
        ])

        # Final output
        if self.training and self.deep_supervision:
            # 直接对中间特征进行监督
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        return self.final(x0_4)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = NestedUNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
