from abc import abstractmethod
from decimal import Decimal
from typing import Union

import numpy as np

from .element import Element
from .element_stateful import ElementStateful


class Observer(Element):
    """观测器。

    参数:
        target_element: 目标元素。
        sampling_period: 采样周期。
        position:
            目标元素中被观测的位置。 `position` 的元素必须为整数或布尔值。若 `position`
            的元素为整数，则 `position` 必须为矩阵，每行为目标元素中被观测的一个位置；若
            `position` 的元素为布尔值，则 `position` 的形状必须与被观测的目标元素相同，
            `position` 中值为 ``True`` 的元素所在位置即目标元素中被观测的位置。

    提示:
        本类是SNN模型中用于观测其他元素的所有类的公共基类。
    """
    def __init__(
            self, target_element: ElementStateful,
            sampling_period: Union[float, Decimal], position: np.ndarray):
        super().__init__()
        if not isinstance(target_element, ElementStateful):
            raise TypeError(
                "target_element must be an ElementStateful instance.")
        if isinstance(sampling_period, Decimal):
            sampling_period_ = sampling_period
        else:
            sampling_period_ = float(sampling_period)
        if sampling_period_ <= 0:
            raise ValueError("sampling_period must be a positive number.")
        position_ = np.array(position)
        if np.issubdtype(position_.dtype, np.integer):
            if position_.ndim != 2:
                raise ValueError(
                    "If position consists of integer elements, it must be a "
                    "matrix.")
        elif np.issubdtype(position_.dtype, np.bool_):
            pass
        else:
            raise ValueError(
                "position must consist of either integer or boolean elements.")
        # 对position的其他检查由派生类实现。Observer类缺乏该检查所需的必要信息。
        self.__target_element: ElementStateful = target_element
        self.__sampling_period: Union[float, Decimal] = sampling_period_
        self.__position: np.ndarray = position_

    def get_target_element(self) -> ElementStateful:
        """获取被观测的目标元素。

        返回值:
            被观测的目标元素。
        """
        return self.__target_element

    def get_sampling_period(self) -> Union[float, Decimal]:
        """获取采样周期。

        返回值:
            采样周期。
        """
        return self.__sampling_period

    def get_position(self) -> np.ndarray:
        """获取目标元素中被观测的位置。

        返回值:
            目标元素中被观测的位置。
        """
        return self.__position
