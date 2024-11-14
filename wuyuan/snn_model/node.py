from collections.abc import Sequence

import numpy as np

from .element import Element
from .utils.node import check_shape, get_neuron_index, get_neuron_position


class Node(Element):
    """脉冲神经网络的节点，可发放脉冲。

    参数:
        shape: 形状。
    """
    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.__shape: tuple[int, ...] = check_shape(shape)

    def get_shape(self) -> tuple[int, ...]:
        """获取节点形状。

        返回值:
            节点的形状。
        """
        return self.__shape

    def get_neuron_index(self, position: np.ndarray = None) -> np.ndarray:
        """获取给定位置的神经元在节点内部的序号。

        参数:
            position:
                神经元的位置或序号。 `position` 是一个整数矩阵或整数向量。若 `position`
                是矩阵，则每行给出一个神经元在节点中的位置；若 `position` 是向量，则每个元
                素给出一个神经元在节点中的序号。

        返回值:
            神经元在节点内部的序号组成的向量。若 `position` 是 ``None`` ，则本方法依次返
            回节点中实际存在的各个神经元在节点内部的序号。否则，本方法返回的各个神经元序号与
            `position` 给出的各个神经元依次对应，对于不存在的神经元，本方法返回的神经元序号
            为-1。
        """
        if position is None:
            return np.arange(np.prod(self.__shape))
        return get_neuron_index(self.__shape, position)

    def get_neuron_position(self, index: np.ndarray = None) -> np.ndarray:
        """获取给定的节点内部的神经元序号在节点中对应的位置。

        参数:
            index: 神经元的序号或位置。 `index` 是一个整数向量或整数矩阵。若 `index` 是
            向量，则每个元素给出一个神经元在节点中的序号；若 `index` 是矩阵，则每行给出一个
            神经元在节点中的位置。

        返回值:
            神经元的位置。本方法返回一个整数矩阵，每行给出一个神经元在节点中的位置。若
            `index` 是 ``None`` ，则本方法依次返回节点中实际存在的各个神经元在节点中的位
            置。否则，本方法返回的各个神经元位置与 `index` 给出的各个神经元依次对应，对于不
            存在的神经元，矩阵中对应行的各个元素均为-1。
        """
        if index is None:
            return \
                np.array(
                    np.unravel_index(
                        np.arange(np.prod(self.__shape)), self.__shape)) \
                .T
        return get_neuron_position(self.__shape, index)
