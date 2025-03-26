"""
UNet model for image segmentation based on the paper:
https://arxiv.org/abs/1505.04597
Nested UNet model for image segmentation based on the paper:
http://arxiv.org/abs/1807.10165
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class FlexibleDoubleConv(nn.Module):
    """
    Flexible Double Convolution for UNet.
    """
    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()
        middle_channels = middle_channels or out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet model for image segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], bilinear=True):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of the UNet
        for feature in features:
            self.downs.append(FlexibleDoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of the UNet
        for feature in reversed(features):
            self.ups.append(
                # use bilinear interpolation to reduce the checkerboard effect
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    # use 1x1 convolution to reduce the number of channels
                    nn.Conv2d(feature * 2, feature, kernel_size=1)
                )
                if bilinear else
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(FlexibleDoubleConv(feature * 2, feature))

        self.bottleneck = FlexibleDoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse the skip connections

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # apply the transpose convolution
            skip_connection = skip_connections[idx // 2]  # get the skip connection
            # if the shape of the x is not the same as the skip connection, resize the x
            if x.shape != skip_connection.shape:
                # take out the height and width of the x
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # apply the double convolution

        return self.final_conv(x)


class NestedUNet(nn.Module):
    """
    Nested UNet (UNet++) model for image segmentation.
    Implements dense skip connections and deep supervision.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256, 512], deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Common layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # First column (down path)
        self.conv0_0 = FlexibleDoubleConv(in_channels, features[0])
        self.conv1_0 = FlexibleDoubleConv(features[0], features[1])
        self.conv2_0 = FlexibleDoubleConv(features[1], features[2])
        self.conv3_0 = FlexibleDoubleConv(features[2], features[3])
        self.conv4_0 = FlexibleDoubleConv(features[3], features[4])

        # First nested layer
        self.conv0_1 = FlexibleDoubleConv(features[0] + features[1], features[0])  # 32 + 64 -> 32
        self.conv1_1 = FlexibleDoubleConv(features[1] + features[2], features[1])  # 64 + 128 -> 64
        self.conv2_1 = FlexibleDoubleConv(features[2] + features[3], features[2])  # 128 + 256 -> 128
        self.conv3_1 = FlexibleDoubleConv(features[3] + features[4], features[3])  # 256 + 512 -> 256

        # Second nested layer
        self.conv0_2 = FlexibleDoubleConv(features[0]*2 + features[1], features[0])  # (32 + 32 + 64) -> 32
        self.conv1_2 = FlexibleDoubleConv(features[1]*2 + features[2], features[1])  # (64 + 64 + 128) -> 64
        self.conv2_2 = FlexibleDoubleConv(features[2]*2 + features[3], features[2])  # (128 + 128 + 256) -> 128

        # Third nested layer
        self.conv0_3 = FlexibleDoubleConv(features[0]*3 + features[1], features[0])  # (32*3 + 64) -> 32
        self.conv1_3 = FlexibleDoubleConv(features[1]*3 + features[2], features[1])  # (64*3 + 128) -> 64

        # Fourth nested layer
        self.conv0_4 = FlexibleDoubleConv(features[0]*4 + features[1], features[0])  # (32*4 + 64) -> 32

        # Final layers for deep supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(features[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(features[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _size_check(self, up_feat, skip_feat):
        """Helper function to check and adjust feature map sizes"""
        if up_feat.shape[2:] != skip_feat.shape[2:]:
            up_feat = TF.resize(up_feat, size=skip_feat.shape[2:], antialias=None)
        return up_feat

    def forward(self, x):
        # Column 0 down path
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested path 1
        x1_0_up = self._size_check(self.up(x1_0), x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0_up], 1))

        x2_0_up = self._size_check(self.up(x2_0), x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0_up], 1))

        x3_0_up = self._size_check(self.up(x3_0), x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0_up], 1))

        x4_0_up = self._size_check(self.up(x4_0), x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0_up], 1))

        # Nested path 2
        x1_1_up = self._size_check(self.up(x1_1), x0_1)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1_up], 1))

        x2_1_up = self._size_check(self.up(x2_1), x1_1)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1_up], 1))

        x3_1_up = self._size_check(self.up(x3_1), x2_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1_up], 1))

        # Nested path 3
        x1_2_up = self._size_check(self.up(x1_2), x0_2)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2_up], 1))

        x2_2_up = self._size_check(self.up(x2_2), x1_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2_up], 1))

        # Nested path 4
        x1_3_up = self._size_check(self.up(x1_3), x0_3)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3_up], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        return self.final(x0_4)


def test():
    # 3 is the number of batch size, 1 is the number of channels, 161 is the height and width
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
