from collections.abc import Iterator, Sequence, Mapping
from typing import Union, Any

import numpy as np

from ..element_model_parameter import ElementModelParameter
from ..topology_model import TopologyModel
from ..type.quantized_numeric_type import QuantizedNumericType
from ..utils.node import get_neuron_position


class OneToOne(TopologyModel):
    def __init__(self, presynaptic_shape: Sequence[int], postsynaptic_shape: Sequence[int], parameter: Mapping[str, Any]):
        """
        构造函数
        Args:
            presynaptic_shape: 突触前神经元形状
            postsynaptic_shape: 突触后神经元形状
        """
        super().__init__(presynaptic_shape, postsynaptic_shape)
        if np.prod(presynaptic_shape) != np.prod(postsynaptic_shape):
            raise ValueError("The number of presynaptic neurons must be equal to the number of postsynaptic neurons.")

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。

        Returns:
            str: 模型的唯一标识符 "OneToOne"。
        """
        return "OneToOne"

    def get_shared_synapse_state_shape(self) -> tuple[int, ...]:
        """
        获取突触状态张量的形状。
        本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Returns:
            共享突触状态张量的形状 type: tuple[int,...]
        """
        return self.get_instantiated_synapse_state_shape()

    def get_instantiated_synapse_state_shape(self) -> tuple[int, ...]:
        """
        获取突触状态张量的形状。
        本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Returns:
            返回的形状取决于具体的拓扑结构模型。
        """
        return (np.prod(self.get_presynaptic_shape()),)

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
        postsynaptic_position = get_neuron_position(self.get_postsynaptic_shape(), postsynaptic_position)

        # 将 postsynaptic_position 转换为一维序号
        flat_indices = np.ravel_multi_index(postsynaptic_position.T, self.get_postsynaptic_shape())
        # 将所有的 flat_indices 转换为对应的 presynaptic_position
        unravel_positions = np.array(np.unravel_index(flat_indices, self.get_presynaptic_shape())).T
        # 将 unravel_positions 变成包含二维矩阵的数组，其中每个矩阵都有 1 行
        presynaptic_positions = unravel_positions[:, np.newaxis, :]

        return presynaptic_positions

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
        presynaptic_position = get_neuron_position(self.get_presynaptic_shape(), presynaptic_position)

        # 将 presynaptic_position 转换为一维序号
        flat_indices = np.ravel_multi_index(presynaptic_position.T, self.get_presynaptic_shape())
        # 将所有的 flat_indices 转换为对应的 postsynaptic_position
        unravel_positions = np.array(np.unravel_index(flat_indices, self.get_postsynaptic_shape())).T
        # 将 unravel_positions 变成包含二维矩阵的数组，其中每个矩阵都有 1 行
        postsynaptic_positions = unravel_positions[:, np.newaxis, :]

        return postsynaptic_positions

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
        return self.get_input_connection_instantiated(postsynaptic_position)

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
        presynaptic_positions = self.get_connected_presynaptic_position(postsynaptic_position)
        synapse_state_positions = np.ravel_multi_index(presynaptic_positions.T, self.get_presynaptic_shape())
        synapse_state_positions = synapse_state_positions[:, np.newaxis]

        return presynaptic_positions, synapse_state_positions

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
        return self.get_output_connection_instantiated(presynaptic_position)

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
        postsynaptic_positions = self.get_connected_postsynaptic_position(presynaptic_position)
        synapse_state_positions = np.ravel_multi_index(postsynaptic_positions.T, self.get_postsynaptic_shape())
        synapse_state_positions = synapse_state_positions[:, np.newaxis]

        return postsynaptic_positions, synapse_state_positions

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
        return self.get_instantiated_synapse_state_position(presynaptic_position, postsynaptic_position)

    def get_shared_synapse_state_position_iterator(self) -> Iterator[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。

        本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。
        """
        return self.get_instantiated_synapse_state_position_iterator()

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
        presynaptic_position = get_neuron_position(self.get_presynaptic_shape(), presynaptic_position)
        postsynaptic_position = get_neuron_position(self.get_postsynaptic_shape(), postsynaptic_position)

        if presynaptic_position.shape[0] != postsynaptic_position.shape[0]:
            raise ValueError("The number of rows in presynaptic_position and postsynaptic_position must be equal.")
        presynaptic_indices = np.ravel_multi_index(presynaptic_position.T, self.get_presynaptic_shape())
        postsynaptic_indices = np.ravel_multi_index(postsynaptic_position.T, self.get_postsynaptic_shape())
        synapse_state_position = np.where(presynaptic_indices == postsynaptic_indices, presynaptic_indices, -1).reshape(-1, 1)

        return synapse_state_position

    def get_instantiated_synapse_state_position_iterator(self) -> Iterator[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。
        本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。
        如果突触前和突触后神经元的位置对应，则返回对应的状态张量位置；否则，返回 -1。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。
        """
        presynaptic_shape = self.get_presynaptic_shape()
        postsynaptic_shape = self.get_postsynaptic_shape()
        synapse_state_positions = range(np.prod(presynaptic_shape))

        for presynaptic_position, postsynaptic_position, synapse_state_position in zip(np.ndindex(presynaptic_shape),
                                                                                       np.ndindex(postsynaptic_shape), synapse_state_positions):
                yield (presynaptic_position, postsynaptic_position, (synapse_state_position,))

    def has_parameter(self, name: str) -> bool:
        """
        检查给定名称是否为有效的参数。

        Args:
            name (str): 要检查的参数名称。

        Returns:
            bool: 如果参数名称有效，则返回 True；否则返回 False。
        """
        return False

    def get_parameter_type(self, name: str) -> Union[type, QuantizedNumericType]:
        """
        获取指定参数的类型。

        Args:
            name (str): 参数名称。

        Raises:
            KeyError: 如果 name 不是有效的参数名称。

        Returns:
            Union[type, QuantizedNumericType]: 参数的类型。
        """
        raise KeyError(f"Parameter name '{name}' is not valid.")

    def get_parameter_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取参数类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 参数名称及其类型的迭代器。
        """
        return iter([])

    def __hash__(self):
        return hash((self._TopologyModel__presynaptic_shape, self._TopologyModel__postsynaptic_shape)) # 修改

    def __eq__(self, other: 'OneToOne') -> bool:
        """
        检查当前对象是否与另一个对象相等。

        Args:
            other (OneToOne): 要比较的另一个对象。

        Returns:
            bool: 如果两个对象相等，则返回 True；否则返回 False。
        """
        if not isinstance(other, OneToOne):
            return False

        return (self._TopologyModel__presynaptic_shape == other._TopologyModel__presynaptic_shape
                and self._TopologyModel__postsynaptic_shape == other._TopologyModel__postsynaptic_shape) # 修改
