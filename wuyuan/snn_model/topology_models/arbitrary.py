from collections.abc import Iterator, Sequence, Mapping
from typing import Union, Any

import numpy as np

from ..element_model_parameter import ElementModelParameter
from ..topology_model import TopologyModel
from ..type.quantized_numeric_type import QuantizedNumericType
from ..utils.node import get_neuron_position


class Arbitrary(TopologyModel):
    __PARAMETERS = {
        "sparse": bool,
        "connected": lambda x: np.asarray(x, dtype=bool)
    }

    def __init__(self, presynaptic_shape: Sequence[int], postsynaptic_shape: Sequence[int], parameter: Mapping[str, Any]):
        """
        构造函数
        Args:
            presynaptic_shape: 突触前神经元形状
            postsynaptic_shape: 突触后神经元形状
        """
        parameter = dict(parameter)

        if 'sparse' not in parameter:
            raise ValueError("Parameter 'sparse' is required.")
        if 'connected' not in parameter:
            raise ValueError("Parameter 'connected' is required.")

        sparse = bool(parameter['sparse'])
        super().__init__(presynaptic_shape, postsynaptic_shape)
        if sparse:
            raise ValueError("Sparse connection is not supported yet.")
        else:
            connected = np.asarray(parameter['connected'], dtype=bool)
            if connected.shape != self.get_presynaptic_shape() + self.get_postsynaptic_shape():
                raise ValueError("The shape of connected must be equal to the sum of presynaptic_shape and postsynaptic_shape.")

        self.set_parameter("sparse", sparse)
        self.set_parameter("connected", connected)

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。

        Returns:
            str: 模型的唯一标识符 "Arbitrary"。
        """
        return "Arbitrary"

    def is_sparse(self) -> bool:
        """
        判断连接是否稀疏。
        Returns:
            bool: 是否稀疏
        """
        return self.get_converted_parameter("sparse")

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
            对于全连接，的确是(突触前神经元个数，突触后神经元个数)。
            对于确定性任意连接，可以是(突触前神经元个数，突触后神经元个数)。
            对于卷积连接，是突触前节点形状+突触后节点形状
        """
        return self.get_presynaptic_shape() + self.get_postsynaptic_shape()

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
        connected = self.get_converted_parameter("connected")

        # 找出 conneted 中与给定突触后神经元相关的子张量
        sub_tensors = connected[(..., *tuple(postsynaptic_position.T))]
        # 将突触前神经元位置填入到对应突触后神经元的位置索引中
        split_tensors = np.split(sub_tensors, sub_tensors.shape[-1], axis=-1)
        result = np.empty(postsynaptic_position.shape[0], dtype=object)
        result[:] = [np.column_stack(np.nonzero(t)[:-1]) for t in split_tensors]

        return result

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
        connected = self.get_converted_parameter("connected")

        # 找出 conneted 中与给定突触前神经元相关的子张量
        sub_tensors = connected[(*tuple(presynaptic_position.T), ...)]
        # 将突触后神经元位置填入到对应突触前神经元的位置索引中
        split_tensors = np.split(sub_tensors, sub_tensors.shape[0], axis=0)
        result = np.empty(presynaptic_position.shape[0], dtype=object)
        result[:] = [np.column_stack(np.nonzero(t)[1:]) for t in split_tensors]

        return result

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
        postsynaptic_position = get_neuron_position(self.get_postsynaptic_shape(), postsynaptic_position)

        # 获取所有与突触后神经元相连的突触前神经元的位置
        presynaptic_positions = self.get_connected_presynaptic_position(postsynaptic_position)

        # 初始化 synapse_state_positions，dtype 为 object
        synapse_state_positions = np.empty(postsynaptic_position.shape[0], dtype=object)

        # 分别为每个突触后神经元生成 synapse_state_position 矩阵
        synapse_state_positions[:] = [
            np.hstack((pre_pos, np.tile(post_pos, (pre_pos.shape[0], 1))))
            for pre_pos, post_pos in zip(presynaptic_positions, postsynaptic_position)
        ]

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
        presynaptic_position = get_neuron_position(self.get_presynaptic_shape(), presynaptic_position)

        # 获取所有与突触前神经元相连的突触后神经元的位置
        postsynaptic_positions = self.get_connected_postsynaptic_position(presynaptic_position)

        # 初始化 synapse_state_positions，dtype 为 object
        synapse_state_positions = np.empty(presynaptic_position.shape[0], dtype=object)

        # 分别为每个突触前神经元生成 synapse_state_position 矩阵
        synapse_state_positions[:] = [
            np.hstack((np.tile(pre_pos, (post_pos.shape[0], 1)), post_pos))
            for pre_pos, post_pos in zip(presynaptic_position, postsynaptic_positions)
        ]

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
        connected = self.get_converted_parameter("connected")

        if presynaptic_position.shape[0] != postsynaptic_position.shape[0]:
            raise ValueError("The number of rows in presynaptic_position and postsynaptic_position must be equal.")
        synapse_state_position = np.hstack((presynaptic_position, postsynaptic_position))
        # 找出connected矩阵中那些为False的位置，并将对应的synapse_state_position行填充为-1
        synapse_state_position[np.logical_not(connected[tuple(synapse_state_position.T)])] = -1

        return synapse_state_position

    def get_instantiated_synapse_state_position_iterator(self) -> Iterator[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。
        本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。
        """
        connected = self.get_converted_parameter("connected")
        for presynaptic_position in np.ndindex(self.get_presynaptic_shape()):
            for postsynaptic_position in np.ndindex(self.get_postsynaptic_shape()):
                if connected[presynaptic_position + postsynaptic_position]:
                    yield (presynaptic_position, postsynaptic_position, presynaptic_position + postsynaptic_position)

    def has_parameter(self, name: str) -> bool:
        """
        检查给定名称是否为有效的参数。

        Args:
            name (str): 要检查的参数名称。

        Returns:
            bool: 如果参数名称有效，则返回 True；否则返回 False。
        """
        name = str(name)
        return name in Arbitrary.__PARAMETERS

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
        name = str(name)
        if name in Arbitrary.__PARAMETERS:
            return Arbitrary.__PARAMETERS[name]
        raise KeyError(f"Parameter name '{name}' is not valid.")

    def get_parameter_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取参数类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 参数名称及其类型的迭代器。
        """
        return iter(Arbitrary.__PARAMETERS.items())

    def __hash__(self) -> int:
        """
        获取对象的哈希值。

        Returns:
            int: 对象的哈希值。
        """
        connected = self.get_converted_parameter("connected")
        hash_value = hash((self.__presynaptic_shape, self.__postsynaptic_shape, self.get_converted_parameter("sparse")))
        hash_value ^= hash(connected.tobytes()) ^ hash(connected.shape) ^ hash(np.count_nonzero(connected))
        return hash_value

    def __eq__(self, other: 'Arbitrary') -> bool:
        """
        检查当前对象是否与另一个对象相等。

        Args:
            other (Arbitrary): 要比较的另一个对象。

        Returns:
            bool: 如果两个对象相等，则返回 True；否则返回 False。
        """
        if not isinstance(other, Arbitrary):
            return False
        return (self.__presynaptic_shape == other.__presynaptic_shape
                and self.__postsynaptic_shape == other.__postsynaptic_shape
                and self.get_converted_parameter("sparse") == other.get_converted_parameter("sparse")
                and np.array_equal(self.get_converted_parameter("connected"), other.get_converted_parameter("connected")))
