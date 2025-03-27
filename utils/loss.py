"""
损失函数模块，定义了多种损失函数，包括二分类交叉熵损失、交叉熵损失、Dice损失和Focal损失。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module, ABC):
    """损失函数基类,定义通用接口和行为"""

    def __init__(self, data_format: str = 'BCHW'):
        super().__init__()
        self.loss_history: List[float] = []
        self.data_format = data_format.upper()

        if self.data_format not in ['BCHW', 'BHWC']:
            raise ValueError(f"Unsupported data format: {data_format}")

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播计算损失值"""
        pass

    def update_history(self, loss_value: float) -> None:
        """记录损失值"""
        self.loss_history.append(loss_value)

    def reset_history(self) -> None:
        """重置损失记录"""
        self.loss_history = []

    def _get_channel_dim(self) -> int:
        """获取通道维度"""
        return 1 if self.data_format == 'BCHW' else -1

    def _get_reduction_dims(self) -> Tuple[int, ...]:
        """获取需要规约的维度"""
        if self.data_format == 'BCHW':
            return (0, 2, 3)  # 规约 batch 和空间维度
        return (0, 1, 2)  # BHWC 格式下规约 BHW 维度

    def _format_target(
        self,
        target: torch.Tensor,
        num_classes: Optional[int] = None,
        multi_class: bool = False
    ) -> torch.Tensor:
        """格式化目标张量

        Args:
            target: 输入张量 (B,H,W)
            num_classes: 类别数(多分类时需要)
            multi_class: 是否为多分类

        Returns:
            格式化后的张量 (B,C,H,W) 或 (B,H,W,C)
        """
        if target.dim() == 3:
            if multi_class and num_classes is not None:
                # one_hot 默认输出 BHWC 格式
                target = F.one_hot(target.long(), num_classes=num_classes)
                if self.data_format == 'BCHW':
                    target = target.permute(0, 3, 1, 2)
            else:
                # 二分类情况
                target = target.unsqueeze(
                    1 if self.data_format == 'BCHW' else -1
                )
        return target


class BinaryCrossEntropyLoss(BaseLoss):
    """二分类交叉熵损失，用于计算二分类任务的损失"""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        data_format: str = 'BCHW'
    ):
        super().__init__(data_format)
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算二分类交叉熵损失

        Args:
            pred: 预测张量 (B,1,H,W) 或 (B,H,W,1)
            target: 目标张量 (B,H,W) 或 (B,1,H,W)

        Returns:
            损失值
        """
        target = self._format_target(target)
        loss = self.bce(pred, target.float())
        self.update_history(loss.item())
        return loss


class CrossEntropyLoss(BaseLoss):
    """交叉熵损失,支持二分类和多分类"""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(pred, target)
        self.update_history(loss.item())
        return loss


class DiceLoss(BaseLoss):
    """Dice损失函数，用于计算分割任务的损失"""

    def __init__(
        self,
        smooth: float = 1e-6,
        multi_class: bool = False,
        ignore_index: Optional[int] = None,
        data_format: str = 'BCHW'
    ):
        """
        Args:
            smooth: 平滑参数
            multi_class: 是否为多分类
            ignore_index: 忽略的类别索引
            data_format: 数据格式，支持'BCHW'或'BHWC'
        """
        super().__init__(data_format)
        self.smooth = smooth
        self.multi_class = multi_class
        self.ignore_index = ignore_index

    def _compute_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dims: Tuple[int, ...]
    ) -> torch.Tensor:
        """计算Dice系数"""
        intersection = torch.sum(pred * target, dim=dims)
        cardinality = torch.sum(pred + target, dim=dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算Dice损失

        Args:
            pred: 预测张量 (B,C,H,W) 或 (B,H,W,C)
            target: 目标张量 (B,H,W)

        Returns:
            损失值
        """
        # 确保输入是概率分布
        channel_dim = self._get_channel_dim()
        if self.multi_class:
            pred = F.softmax(pred, dim=channel_dim)
        else:
            pred = torch.sigmoid(pred)

        # 格式化目标张量
        target = self._format_target(
            target,
            pred.shape[channel_dim],
            self.multi_class
        )

        # 处理需要忽略的类别
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask
            target = target * mask

        # 计算Dice系数
        dims = self._get_reduction_dims()
        dice = self._compute_dice(pred, target.float(), dims)

        if self.multi_class:
            dice = dice.mean()

        loss = 1 - dice
        self.update_history(loss.item())
        return loss


class FocalLoss(BaseLoss):
    """Focal Loss,用于解决类别不平衡问题"""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        multi_class: bool = False,
        data_format: str = 'BCHW'
    ):
        """
        Args:
            alpha: 正样本权重系数
            gamma: 聚焦参数
            reduction: 降维方式，'mean' 或 'sum'
            multi_class: 是否为多分类
            data_format: 数据格式，支持'BCHW'或'BHWC'
        """
        super().__init__(data_format)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.multi_class = multi_class

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Focal Loss

        Args:
            pred: 预测张量
                二分类: (B,1,H,W)/(B,H,W,1)
                多分类: (B,C,H,W)/(B,H,W,C)
            target: 目标张量
                二分类: (B,H,W) 值为 0/1
                多分类: (B,H,W) 值为类别索引
        """
        if self.multi_class:
            # 多分类情况
            ce_loss = F.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            # 二分类情况
            target = self._format_target(target)

            bce_loss = F.binary_cross_entropy_with_logits(
                pred, target.float(), reduction='none'
            )
            pt = torch.exp(-bce_loss)

            # 对正负样本分别加权
            alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_weight * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        self.update_history(focal_loss.item())
        return focal_loss
