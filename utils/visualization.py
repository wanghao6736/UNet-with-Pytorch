"""
可视化工具模块，定义了可视化工具基类和损失曲线、评估指标可视化工具。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class BaseVisualizer(ABC):
    """可视化工具基类"""

    def __init__(
        self,
        title: str,
        figsize: Tuple[int, int] = (10, 5),
        xlabel: str = '',
        ylabel: str = '',
        dpi: int = 300
    ):
        self.title = title
        self.figsize = figsize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig = None
        self.ax = None
        self.dpi = dpi

    def initialize_plot(self) -> None:
        """初始化绘图"""
        if self.fig is not None:
            self.close_plot()
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._set_labels()

    def _set_labels(self) -> None:
        """设置图表标签"""
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True)

    def save_plot(self, save_path: Union[str, Path], transparent: bool = False) -> None:
        """保存图表

        Args:
            save_path: 保存路径
            transparent: 是否透明背景
        """
        if self.fig is None:
            raise ValueError("Plot not initialized")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, transparent=transparent)

    def show_plot(self) -> None:
        """显示图表"""
        if self.fig is None:
            raise ValueError("Plot not initialized")
        plt.show(block=False)
        plt.pause(0.1)

    def close_plot(self) -> None:
        """关闭图表并重置状态"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    @abstractmethod
    def _plot_data(self, data, **kwargs) -> None:
        """绘制数据"""
        pass

    def plot(
        self,
        data,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        close: bool = True,
        **kwargs
    ) -> None:
        """绘图并控制显示

        Args:
            data: 绘图数据
            save_path: 保存路径
            show: 是否显示, 显示后无法使用 plt.savefig() 保存图片
            close: 是否关闭图表, 关闭后无法使用 plt.show() 显示图片
            **kwargs: 其他参数
        """
        self.initialize_plot()
        self._plot_data(data, **kwargs)

        if save_path:
            self.save_plot(save_path)

        if show:
            self.show_plot()

        if close:
            self.close_plot()


class LossVisualizer(BaseVisualizer):
    """损失曲线可视化工具"""

    def __init__(
        self,
        title: str = "Training Loss",
        figsize: Tuple[int, int] = (10, 6)
    ):
        super().__init__(
            title=title,
            figsize=figsize,
            xlabel='Iteration',
            ylabel='Loss'
        )

    def _plot_data(self, loss_values: Union[List[float], np.ndarray]) -> None:
        self.ax.plot(loss_values, label='loss')
        self.ax.legend()


class MetricsVisualizer(BaseVisualizer):
    """评估指标可视化工具"""

    def __init__(
        self,
        title: str = 'Metrics Scores',
        figsize: Tuple[int, int] = (10, 5),
        xlabel: str = 'Class',
        ylabel: str = 'Score'
    ):
        super().__init__(
            title=title,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel
        )

    def _plot_data(self, data, **kwargs) -> None:
        """统一的绘图接口

        Args:
            data: 可以是 List[float] 或 Dict[str, float]
            **kwargs: 额外的绘图参数
        """
        if isinstance(data, list):
            self._plot_list_data(data)
        elif isinstance(data, dict):
            self._plot_dict_data(data,
                                 mean_color=kwargs.get('mean_color', 'r'),
                                 mean_style=kwargs.get('mean_style', '--'))
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    def _plot_list_data(self, metric: List[float]) -> None:
        """绘制列表类型的数据"""
        self.ax.plot(metric, label='metric')
        self.ax.legend()

    def _plot_dict_data(self, scores: Dict[str, float],
                        mean_color: str = 'r',
                        mean_style: str = '--') -> None:
        """绘制字典类型的数据"""
        # 分离均值和类别分数
        mean = scores.get('mean', None)
        classes = [k for k in scores.keys() if k != 'mean']
        values = [scores[k] for k in classes]

        # 绘制柱状图
        self.ax.bar(classes, values)

        # 绘制均值线
        if mean is not None:
            self.ax.axhline(
                y=mean,
                color=mean_color,
                linestyle=mean_style,
                label='Mean'
            )
            self.ax.legend()


def plot_dice_scores(
    scores: Dict[str, float],
    title: str = 'Dice Scores by Class'
) -> None:
    """绘制Dice分数分布(便捷函数)"""
    visualizer = MetricsVisualizer(title=title)
    visualizer.plot(scores)
