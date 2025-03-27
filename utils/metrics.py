"""
评估指标模块，定义了多种评估指标，包括Dice系数、IoU、准确率等。
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class BaseMetric(ABC):
    """评估指标基类"""

    def __init__(self, data_format: str = 'BCHW'):
        self.data_format = data_format.upper()
        if self.data_format not in ['BCHW', 'BHWC']:
            raise ValueError(f"Unsupported data format: {data_format}")
        self.reset()

    @abstractmethod
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """更新评估指标"""
        pass

    @abstractmethod
    def compute(self) -> float:
        """计算累积结果"""
        pass

    def reset(self) -> None:
        """重置评估指标"""
        pass

    def _get_channel_dim(self) -> int:
        """获取通道维度"""
        return 1 if self.data_format == 'BCHW' else -1

    def _get_reduction_dims(self) -> Tuple[int, ...]:
        """获取需要规约的维度"""
        if self.data_format == 'BCHW':
            return (0, 2, 3)  # 规约 batch 和空间维度
        return (0, 1, 2)  # BHWC 格式下规约 BHW 维度


class MetricBase(BaseMetric):
    """评估指标通用基类"""

    def __init__(
        self,
        multi_class: bool = False,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5,
        data_format: str = 'BCHW'
    ):
        super().__init__(data_format)
        self.multi_class = multi_class
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.class_scores = {}  # 每个类别的分数
        self.metric_history = []  # 每个类别的评估指标历史

    def update_history(self, metric: float) -> None:
        """更新评估指标历史"""
        self.metric_history.append(metric)

    def reset_history(self) -> None:
        """重置评估指标历史"""
        self.metric_history = []

    def _preprocess_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        apply_mask: bool = True,
        one_hot_target: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """预处理输入数据

        Args:
            pred: 预测张量 (B,C,H,W) 或 (B,H,W,C)
            target: 目标张量 (B,H,W)
            apply_mask: 是否应用mask
            one_hot_target: 是否将target转换为one-hot形式

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            处理后的预测、目标张量和mask(如果有ignore_index)
        """
        with torch.no_grad():
            channel_dim = self._get_channel_dim()
            if self.multi_class:
                pred = F.softmax(pred, dim=channel_dim)
                if one_hot_target and target.dim() == 3:
                    target = F.one_hot(
                        target.long(),
                        num_classes=pred.shape[channel_dim]
                    )
                    if self.data_format == 'BCHW':
                        target = target.permute(0, 3, 1, 2)
            else:
                pred = torch.sigmoid(pred)
                pred = (pred > self.threshold).float()

                # 处理target维度
                if target.dim() == 3:
                    target = target.unsqueeze(
                        1 if self.data_format == 'BCHW' else -1
                    )

            mask = (target != self.ignore_index) if self.ignore_index is not None else None
            if mask is not None and apply_mask:
                pred = pred * mask
                target = target * mask

            return pred, target, mask

    def _update_class_scores(self, scores: torch.Tensor) -> None:
        """更新每个类别的分数

        Args:
            scores: 每个类别的分数张量
        """
        if self.multi_class:
            for i, score in enumerate(scores):
                if i not in self.class_scores:
                    self.class_scores[i] = []
                self.class_scores[i].append(score.item())

    def _compute_class_means(self) -> Dict[str, float]:
        """计算每个类别的平均分数

        Returns:
            Dict[str, float]: 每个类别的平均分数
        """
        class_means = {
            f'class_{i}': sum(scores) / len(scores)
            for i, scores in self.class_scores.items()
        }
        return class_means


class DiceScore(MetricBase):
    """Dice系数评估指标"""

    def __init__(
        self,
        smooth: float = 1e-6,
        multi_class: bool = False,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5,
        data_format: str = 'BCHW'
    ):
        super().__init__(multi_class, ignore_index, threshold, data_format)
        self.smooth = smooth
        self.reset()

    def reset(self) -> None:
        self.total_dice = 0.0
        self.count = 0
        self.class_scores = {}

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """更新Dice分数

        Args:
            pred: 预测张量 (B,C,H,W) 或 (B,H,W,C)
            target: 目标张量 (B,H,W)

        Returns:
            float: 当前批次的Dice分数
        """
        pred, target, mask = self._preprocess_inputs(pred, target)

        # 计算Dice系数
        dims = self._get_reduction_dims()
        intersection = torch.sum(pred * target, dim=dims)
        cardinality = torch.sum(pred + target, dim=dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # 更新统计
        self._update_class_scores(dice)
        dice_value = dice.mean().item() if self.multi_class else dice.item()
        self.total_dice += dice_value
        self.count += 1
        self.update_history(dice_value)

        return dice_value

    def compute(self) -> Union[float, Dict[str, float]]:
        """计算累积的评估结果

        Returns:
            float: 平均Dice分数
            或
            Dict[str, float]: 每个类别的Dice分数
        """
        if not self.multi_class:
            return self.total_dice / max(self.count, 1)

        class_means = self._compute_class_means()
        class_means['mean'] = self.total_dice / max(self.count, 1)
        return class_means


class IoU(MetricBase):
    """IoU(Intersection over Union)评估指标"""

    def __init__(
        self,
        smooth: float = 1e-6,
        multi_class: bool = False,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5,
        data_format: str = 'BCHW'
    ):
        super().__init__(multi_class, ignore_index, threshold, data_format)
        self.smooth = smooth
        self.reset()

    def reset(self) -> None:
        self.total_iou = 0.0
        self.count = 0
        self.class_scores = {}

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """更新IoU分数

        Args:
            pred: 预测张量 (B,C,H,W) 或 (B,H,W,C)
            target: 目标张量 (B,H,W)

        Returns:
            float: 当前批次的IoU分数
        """
        pred, target, mask = self._preprocess_inputs(pred, target)

        # 计算IoU
        dims = self._get_reduction_dims()
        intersection = torch.sum(pred * target, dim=dims)
        union = torch.sum(pred + target, dim=dims) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        # 更新统计
        self._update_class_scores(iou)
        iou_value = iou.mean().item() if self.multi_class else iou.item()
        self.total_iou += iou_value
        self.count += 1
        self.update_history(iou_value)
        return iou_value

    def compute(self) -> Union[float, Dict[str, float]]:
        """计算累积的评估结果

        Returns:
            float: 平均IoU分数
            或
            Dict[str, float]: 每个类别的IoU分数
        """
        if not self.multi_class:
            return self.total_iou / max(self.count, 1)

        class_means = self._compute_class_means()
        class_means['mean'] = self.total_iou / max(self.count, 1)
        return class_means


class Accuracy(MetricBase):
    """准确率评估指标"""

    def __init__(
        self,
        multi_class: bool = False,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5,
        weighted: bool = False,
        data_format: str = 'BCHW'
    ):
        super().__init__(multi_class, ignore_index, threshold, data_format)
        self.weighted = weighted
        self.reset()

    def reset(self) -> None:
        self.total_correct = 0
        self.total_pixels = 0
        self.class_scores = {}

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """更新准确率分数

        Args:
            pred: 预测张量 (B,C,H,W) 或 (B,H,W,C)
            target: 目标张量 (B,H,W)

        Returns:
            float: 当前批次的准确率
        """
        pred, target, mask = self._preprocess_inputs(pred, target, apply_mask=False, one_hot_target=False)

        if self.multi_class:
            pred = pred.argmax(dim=self._get_channel_dim())

        # 应用mask(如果存在)
        valid_pixels = target.numel()
        if mask is not None:
            pred = pred * mask
            target = target * mask
            valid_pixels = mask.sum().item()

        correct = (pred == target).float()

        if self.weighted and self.multi_class:
            unique_classes = torch.unique(target)
            class_weights = {}
            for class_idx in unique_classes:
                if class_idx == self.ignore_index:
                    continue
                class_mask = target == class_idx
                class_weights[class_idx.item()] = 1.0 / class_mask.sum().item()

            # 归一化权重
            total_weight = sum(class_weights.values())
            class_weights = {k: v / total_weight for k, v in class_weights.items()}

            # 应用权重
            weighted_correct = torch.zeros_like(correct)
            for class_idx, weight in class_weights.items():
                class_mask = target == class_idx
                weighted_correct += correct * class_mask * weight
            correct = weighted_correct

        # 更新统计
        self.total_correct += correct.sum().item()
        self.total_pixels += valid_pixels

        # 计算当前批次的准确率
        accuracy = correct.sum().item() / valid_pixels

        # 更新类别分数
        if self.multi_class:
            unique_classes = torch.unique(target)
            for class_idx in unique_classes:
                if class_idx == self.ignore_index:
                    continue
                class_mask = target == class_idx
                class_correct = (correct * class_mask).sum().item()
                class_total = class_mask.sum().item()
                class_acc = class_correct / class_total if class_total > 0 else 0

                if class_idx.item() not in self.class_scores:
                    self.class_scores[class_idx.item()] = []
                self.class_scores[class_idx.item()].append(class_acc)

        self.update_history(accuracy)

        return accuracy

    def compute(self) -> Union[float, Dict[str, float]]:
        """计算累积的评估结果

        Returns:
            float: 平均准确率
            或
            Dict[str, float]: 每个类别的准确率
        """
        if not self.multi_class:
            return self.total_correct / max(self.total_pixels, 1)

        class_means = self._compute_class_means()
        class_means['mean'] = self.total_correct / max(self.total_pixels, 1)
        return class_means


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """单次计算Dice分数的便捷函数"""
    dice_calculator = DiceScore(smooth=smooth)
    return dice_calculator.update(pred, target)


def compute_iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> float:
    """单次计算IoU分数的便捷函数"""
    iou_calculator = IoU(smooth=smooth)
    return iou_calculator.update(pred, target)


def compute_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighted: bool = False
) -> float:
    """单次计算准确率的便捷函数"""
    acc_calculator = Accuracy(weighted=weighted)
    return acc_calculator.update(pred, target)
