"""
用于提取patches和重建图像的工具类, 可以用于提取正方形patches和重建图像, 并可视化patches和重建图像
"""
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class PatchExtractor:
    def __init__(self, patch_size: int, overlap: int):
        """
        初始化patch提取器

        Args:
            patch_size: patch的边长(正方形)
            overlap: 相邻patch之间的重叠像素数
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.rows = None
        self.cols = None

    def _calculate_grid_size(self, height: int, width: int) -> Tuple[int, int]:
        """
        计算给定图像尺寸下的patches网格大小，确保覆盖整个图像

        Args:
            height: 图像高度
            width: 图像宽度

        Returns:
            rows: 行数
            cols: 列数
        """
        # 计算需要的行列数，向上取整以确保覆盖所有区域
        rows = (height + self.stride - 1) // self.stride
        cols = (width + self.stride - 1) // self.stride

        # 确保最后一个patch不会超出图像范围
        if (rows - 1) * self.stride + self.patch_size > height:
            rows = max(1, (height - self.patch_size + self.stride - 1) // self.stride + 1)
        if (cols - 1) * self.stride + self.patch_size > width:
            cols = max(1, (width - self.patch_size + self.stride - 1) // self.stride + 1)
        return rows, cols

    def extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """
        从图像中提取patches

        Args:
            image: 输入图像张量 [C, H, W]

        Returns:
            patches张量 [N, C, patch_size, patch_size]
        """
        C, H, W = image.shape

        # 计算网格大小
        self.rows, self.cols = self._calculate_grid_size(H, W)
        self.original_size = (H, W)

        # 创建填充后的图像
        pad_height = (self.rows - 1) * self.stride + self.patch_size
        pad_width = (self.cols - 1) * self.stride + self.patch_size
        self.padding = (pad_height, pad_width)

        # 计算需要的填充量
        pad_bottom = max(0, pad_height - H)
        pad_right = max(0, pad_width - W)

        # 对图像进行填充
        if pad_bottom > 0 or pad_right > 0:
            padded_image = F.pad(image, (0, pad_right, 0, pad_bottom), mode='reflect')
        else:
            padded_image = image

        # 使用unfold提取patches
        patches = padded_image.unfold(1, self.patch_size, self.stride)\
            .unfold(2, self.patch_size, self.stride)\
            .reshape(C, -1, self.patch_size, self.patch_size)\
            .permute(1, 0, 2, 3)

        return patches

    def reconstruct_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        从patches重建原始图像

        Args:
            patches: patches张量 [N, C, patch_size, patch_size]

        Returns:
            重建后的图像张量 [C, H, W]
        """
        if self.rows is None or self.cols is None:
            raise ValueError("Please run extract_patches first")

        N, C, H, W = patches.shape
        orig_H, orig_W = self.original_size
        pad_height, pad_width = self.padding

        # 创建fold层，使用填充后的尺寸
        fold = torch.nn.Fold(
            output_size=(pad_height, pad_width),
            kernel_size=(self.patch_size, self.patch_size),
            stride=self.stride
        )

        # 重塑patches用于fold操作
        patches_reshaped = patches.permute(1, 2, 3, 0)\
            .reshape(1, C * self.patch_size * self.patch_size, -1)

        # 创建权重tensor用于平均重叠区域
        ones = torch.ones_like(patches)
        ones_reshaped = ones.permute(1, 2, 3, 0)\
            .reshape(1, C * self.patch_size * self.patch_size, -1)

        # 执行fold操作
        output, weights = fold(patches_reshaped), fold(ones_reshaped)

        # 处理权重为0的情况
        weights = torch.where(weights == 0, torch.ones_like(weights), weights)
        output = output / weights

        # 裁剪回原始尺寸，确保不包含填充部分
        output = output[:, :, :orig_H, :orig_W]

        return output.squeeze(0)

    def interleave_patches(self, patches: torch.Tensor, mask_patches: torch.Tensor, cols: int = None) -> torch.Tensor:
        """
        使用reshape和permute交错排列原图和掩码patches

        Args:
            patches: 原图patches [N, 3, H, W]
            mask_patches: 掩码patches [N, 1, H, W]
            cols: 每行显示的patch数量

        Returns:
            交错排列后的patches [2N, 3, H, W]
        """
        N, C, H, W = patches.shape
        mask_patches_3ch = mask_patches.repeat(1, C, 1, 1)

        # 将patches重排为[rows, cols, C, H, W]形状
        cols = cols or self.cols
        rows = (N + cols - 1) // cols

        # 1. 先reshape为[rows, cols, C, H, W]
        patches_reshaped = patches.reshape(rows, cols, C, H, W)
        masks_reshaped = mask_patches_3ch.reshape(rows, cols, C, H, W)

        # 2. 堆叠原图和掩码 [rows, 2, cols, C, H, W]
        stacked = torch.stack([patches_reshaped, masks_reshaped], dim=1)

        # 3. 调整维度顺序并重塑为最终形状 [2*rows*cols, C, H, W]
        return stacked.reshape(-1, C, H, W)

    def _imshow(self, ax: plt.Axes, image: Any, title: str = None, **kwargs):
        """
        显示图像
        """
        ax.imshow(image, **kwargs)
        ax.axis('off')
        if title:
            ax.set_title(title)

    def show_patches(self, patches: torch.Tensor, cols: int = None):
        """
        显示提取的patches

        Args:
            patches: patches张量 [N, C, H, W]
            cols: 显示的行数,如果为None则自动计算
        """
        n_patches = patches.shape[0]
        cols = cols or self.cols
        rows = max(self.rows, int(np.ceil(n_patches / cols)))

        fig, axes = plt.subplots(rows, cols, figsize=(14, int(14 * rows / cols)))
        axes = axes.ravel()

        for idx in range(n_patches):
            if patches.shape[1] == 1:  # 灰度图
                self._imshow(axes[idx], patches[idx, 0], title=f'Patch {idx}', cmap='gray', vmin=0, vmax=1)
            else:  # RGB图
                self._imshow(axes[idx], patches[idx].permute(1, 2, 0), title=f'Patch {idx}')

        plt.tight_layout()
        plt.show()

    def show_comparison(self, image: torch.Tensor, reconstructed: torch.Tensor):
        """
        显示原始图像和重建图像的对比

        Args:
            image: 原始图像张量 [C, H, W]
            reconstructed: 重建图像张量 [C, H, W]
        """
        # 转换为numpy数组并调整维度顺序
        orig_img = image.permute(1, 2, 0).numpy()  # [H, W, C]
        recon_img = reconstructed.permute(1, 2, 0).numpy()  # [H, W, C]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        self._imshow(ax1, orig_img, title='Original Image')
        self._imshow(ax2, recon_img, title='Reconstructed Image')

        plt.tight_layout()
        plt.show()

        mse = torch.mean((image - reconstructed) ** 2)
        print(f"Reconstruction MSE: {mse:.8f}")
