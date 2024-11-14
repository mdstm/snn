from abc import abstractmethod
from typing import Sequence, Iterator

import numpy as np

from .element_model import ElementModel
from .element_model_parameter import ElementModelParameter
from .utils.node import check_shape


class TopologyModel(ElementModel, ElementModelParameter):
    def __init__(self, presynaptic_shape: Sequence[int], postsynaptic_shape: Sequence[int]):
        """
        构造函数
        Args:
            presynaptic_shape:
            postsynaptic_shape:
        """
        ElementModel.__init__(self)
        ElementModelParameter.__init__(self)
        self.__presynaptic_shape: tuple[int, ...] = check_shape(presynaptic_shape)
        self.__postsynaptic_shape: tuple[int, ...] = check_shape(postsynaptic_shape)

    def get_presynaptic_shape(self) -> tuple[int, ...]:
        """

        Returns:
            突触前神经元形状 type: tuple[int,...]

        """
        return self.__presynaptic_shape

    def get_postsynaptic_shape(self) -> tuple[int, ...]:
        """

        Returns:
            突触后神经元形状 type: tuple[int,...]

        """
        return self.__postsynaptic_shape

    @abstractmethod
    def get_shared_synapse_state_shape(self) -> tuple[int, ...]:
        """
        获取突触状态张量的形状。

        Returns:
            共享突触状态张量的形状 type: tuple[int,...]

        """
        raise NotImplementedError()

    @abstractmethod
    def get_instantiated_synapse_state_shape(self) -> tuple[int, ...]:
        """
        获取突触状态张量的形状。

        本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Returns:
            返回的形状取决于具体的拓扑结构模型。
            对于全连接，的确是(突触前神经元个数，突触后神经元个数)。
            对于确定性任意连接，可以是(突触前神经元个数，突触后神经元个数)。
            对于卷积连接，是突触前节点形状+突触后节点形状


        """
        raise NotImplementedError()

    @abstractmethod
    def get_connected_presynaptic_position(self, postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取与给定突触后神经元相连的突触前神经元。

        Args:
            postsynaptic_position: 矩阵，其中每行 postsynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            presynaptic_position: 每个元素 presynaptic_position[i] 是一个矩阵
            每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，该突触前神经元与 postsynaptic_position[i] 表示的突触后神经元相连

        """
        raise NotImplementedError()

    @abstractmethod
    def get_connected_postsynaptic_position(self, presynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取与给定突触前神经元相连的突触后神经元。
        Args:
            presynaptic_position:矩阵，其中每行 presynaptic_position[i] 表示一个突触前神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            postsynaptic_position: 每个元素 postsynaptic_position[i] 是一个矩阵
            每行 postsynaptic_position[i][j] 是一个突触后神经元在其节点中的位置，该突触后神经元与 presynaptic_position[i] 表示的突触前神经元相连。

        """
        raise NotImplementedError()

    @abstractmethod
    def get_input_connection_shared(self, postsynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触后神经元的输入连接。本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Args:
            postsynaptic_position:矩阵，其中每行 postsynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            presynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            presynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 postsynaptic_position[i] 表示的突触后神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        raise NotImplementedError()

    @abstractmethod
    def get_input_connection_instantiated(self, postsynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触后神经元的输入连接。本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量

        Args:
            postsynaptic_position:矩阵，其中每行 postsynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            presynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            presynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 postsynaptic_position[i] 表示的突触后神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_connection_shared(self, presynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触前神经元的输出连接。本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Args:
            presynaptic_position:矩阵，其中每行 presynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            postsynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            postsynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 presynaptic_position[i] 表示的突触前神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_connection_instantiated(self, presynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触前神经元的输出连接。本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Args:
            presynaptic_position:矩阵，其中每行 presynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            postsynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            postsynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 presynaptic_position[i] 表示的突触前神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        raise NotImplementedError()

    @abstractmethod
    def get_shared_synapse_state_position(self, presynaptic_position: np.ndarray,
                                          postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取给定突触的状态值在连接组突触状态张量中的位置。

        Args:
            presynaptic_position: 有n行,每行表示一个突触前神经元的位置
            postsynaptic_position: 有n行,每行表示一个突触后神经元的位置

        Returns:
            synapse_state_position: 有n行,每行表示一个突触在状态张量中的位置

        """
        raise NotImplementedError()

    @abstractmethod
    def get_shared_synapse_state_position_iterator(self) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。

        本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。

        """
        raise NotImplementedError()

    @abstractmethod
    def get_instantiated_synapse_state_position(self, presynaptic_position: np.ndarray,
                                                postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取给定突触的状态值在连接组突触状态张量中的位置。

        Args:
            presynaptic_position: 有n行,每行表示一个突触前神经元的位置
            postsynaptic_position: 有n行,每行表示一个突触后神经元的位置

        Returns:
            synapse_state_position: 有n行,每行表示一个突触在状态张量中的位置


        """
        raise NotImplementedError()

    @abstractmethod
    def get_instantiated_synapse_state_position_iterator(self) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。
        本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。

        """
        raise NotImplementedError()
