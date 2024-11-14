from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from typing import Any, Optional, Union

import numpy as np

from .element_stateful import ElementStateful
from .neuron_group import NeuronGroup
from .node import Node
from .synapse_model import SynapseModel
from .topology_model import TopologyModel


class ConnectionGroup(ElementStateful):
    """连接组

    参数:
        presynaptic_node: 突触前节点。
        postsynaptic_node: 突触后节点。
        topology_model: 拓扑结构模型。
        synapse_model: 突触模型。
        initial_synapse_state_value: 突触状态初值。
        delay: 各个连接共同的延时。
    """
    def __init__(
        self,
        presynaptic_node: Node,
        postsynaptic_node: NeuronGroup,
        topology_model: TopologyModel,
        synapse_model: SynapseModel,
        initial_synapse_state_value: Optional[Mapping[str, tuple[np.ndarray, bool]]] = None,
        delay: Optional[Union[float, Decimal]] = None,
    ):
        if not isinstance(presynaptic_node, Node):
            raise TypeError("presynaptic_node must be an instance of Node.")
        if not isinstance(postsynaptic_node, NeuronGroup):
            raise TypeError("postsynaptic_node must be an instance of NeuronGroup.")
        if not isinstance(topology_model, TopologyModel):
            raise TypeError("topology_model must be an instance of TopologyModel.")
        if not isinstance(synapse_model, SynapseModel):
            raise TypeError("synapse_model must be an instance of SynapseModel.")

        initial_synapse_state_value_ = {}
        if initial_synapse_state_value is not None:
            for key, value in dict(initial_synapse_state_value).items():
                key, value = str(key), tuple(value)
                if key in initial_synapse_state_value_:
                    raise ValueError(f"duplicate synapse state name '{key}' found in initial_synapse_state_value.")
                synapse_state_tensor = synapse_model.check_state_value(key, value[0])
                expected_shape = topology_model.get_shared_synapse_state_shape()
                if synapse_state_tensor.shape != expected_shape:
                    raise ValueError(f"shape of the synapse state value '{key}' mismatches the topology model: "
                                     f"expected shape {expected_shape}, but got {synapse_state_tensor.shape}.")
                initial_synapse_state_value_[key] = (synapse_state_tensor, bool(value[1]))

        if delay is None:
            delay_ = 0.0
        elif isinstance(delay, Decimal):
            delay_ = delay
        else:
            delay_ = float(delay)
        if delay_ < 0:
            raise ValueError("delay must be non-negative.")

        self.__presynaptic_node: Node = presynaptic_node
        self.__postsynaptic_node: Node = postsynaptic_node
        self.__topology_model: TopologyModel = topology_model
        self.__synapse_model: SynapseModel = synapse_model
        self.__initial_synapse_state_value: dict[str, tuple[np.ndarray, bool]] = initial_synapse_state_value_
        self.__delay: Union[float, Decimal] = delay_

    def get_presynaptic_node(self) -> Node:
        """获取突触前节点。

        返回值:
            突触前节点。
        """
        return self.__presynaptic_node

    def get_postsynaptic_node(self) -> Node:
        """获取突触后节点。

        返回值:
            突触后节点。
        """
        return self.__postsynaptic_node

    def get_topology_model(self) -> TopologyModel:
        """获取拓扑结构模型。

        返回值:
            拓扑结构模型。
        """
        return self.__topology_model

    def get_synapse_model(self) -> SynapseModel:
        """获取突触模型。

        返回值:
            突触模型。
        """
        return self.__synapse_model

    def set_initial_synapse_state_value_tensor(self, synapse_state_name: str, synapse_state_value: np.ndarray,
                                               constant: bool = False):
        """设置给定突触状态名称对应的突触状态初值张量。

        参数:
            synapse_state_name: 突触状态名。
            synapse_state_value: 突触状态值。
            constant: 突触状态是否改变的标志位。

        抛出:
            KeyError: `synapse_state_name` 不是突触模型的状态名称。
            ValueError: `synapse_state_value` 的形状不符合拓扑结构模型的限制。
        """
        synapse_state_value_ = self.__synapse_model.check_state_value(synapse_state_name, synapse_state_value)
        expected_shape = self.__topology_model.get_shared_synapse_state_shape()
        if synapse_state_value_.shape != expected_shape:
            raise ValueError("shape of the synapse state value mismatches the topology model: "
                             f"expected shape {expected_shape}, but got {synapse_state_value_.shape}.")
        self.__initial_synapse_state_value[synapse_state_name] = (synapse_state_value_, bool(constant))

    def has_initial_synapse_state_value_tensor(self, synapse_state_name: str) -> bool:
        """判断连接组是否具有给定突触状态名称对应的突触状态初值张量。

        参数:
            synapse_state_name: 突触状态名。

        返回值:
            连接组是否具有给定突触状态名称对应的突触状态初值张量。
        """
        return synapse_state_name in self.__initial_synapse_state_value

    def get_initial_synapse_state_value_tensor(self, synapse_state_name: str) -> np.ndarray:
        """获取给定突触状态名称对应的突触状态初值张量。

        参数:
            synapse_state_name: 突触状态名。

        返回值:
            给定突触状态名称对应的突触状态初值张量。

        抛出:
            KeyError: `synapse_state_name` 不是突触模型的状态名称。
        """
        if not self.has_initial_synapse_state_value_tensor(synapse_state_name):
            raise KeyError(f"synapse state name '{synapse_state_name}' is invalid.")
        return self.__initial_synapse_state_value[synapse_state_name][0]

    def get_initial_synapse_state_value_tensor_iterator(self) -> Iterator[tuple[str, np.ndarray]]:
        """获取突触状态初值张量迭代器。

        返回值:
            每次迭代依次返回突触状态名称和对应的突触状态初值张量。
        """
        for name, (synapse_state_value, _) in self.__initial_synapse_state_value.items():
            yield name, synapse_state_value

    def is_synapse_state_constant(self, synapse_state_name: str) -> bool:
        """判断给定突触状态是否为常量，即，在SNN的运行过程中是否保持不变。

        参数:
            synapse_state_name: 突触状态名。

        返回值:
            给定突触状态是否为常量。
        """
        if not self.has_initial_synapse_state_value_tensor(synapse_state_name):
            raise KeyError(f"Synapse state name '{synapse_state_name}' is invalid.")
        return self.__initial_synapse_state_value[synapse_state_name][1]

    def get_shared_synapse_state_shape(self) -> Sequence[int]:
        """获取突触状态张量的形状。

        返回值:
            突触状态张量的形状。

        提示:
            本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。
        """
        return self.__topology_model.get_shared_synapse_state_shape()

    def get_shared_synapse_state_position(self, presynaptic_position: np.ndarray,
                                          postsynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """获取给定突触的状态值在连接组突触状态张量中的位置。

        参数:
            presynaptic_position: 二维整数矩阵，每行表示一个突触前神经元在其神经元组中的位置。
            postsynaptic_position: 二维整数矩阵，每行表示一个突触后神经元在其神经元组中的位置。

        返回值:
            返回值依次为 `synapse_exists` 和 `synapse_state_position` 。

            `synapse_exists` 为向量，`synapse_exists[i]` 指示第i个前后神经元对间是否存在突触。

            `synapse_state_position` 为二维矩阵。 `presynaptic_position` 和 `postsynaptic_position` 对应位置上的行，
            即， `presynaptic_position[i]` 和 `postsynaptic_position[i]` 共同表示一个突触。
            当且仅当该突触存在， `synapse_exists[i]` 为真，该突触的状态值在状态张量中的位置为 `synapse_state_position[i]` 。

        提示:
            1. 本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

            2. 本方法假设连接组的突触状态张量不是标量。
        """
        synapse_state_position = self.__topology_model.get_shared_synapse_state_position(
            presynaptic_position, postsynaptic_position)
        synapse_exists = (synapse_state_position[:, 0] >= 0)
        return synapse_exists, synapse_state_position

    def get_shared_synapse_state_position_iterator(self) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """获取突触状态值位置迭代器。

        返回值:
            每次迭代一个突触，返回值依次包括：

            1. 突触前神经元在其神经元组中的位置；

            2. 突触后神经元在其神经元组中的位置；

            3. 突触状态值在突触状态张量中的位置。

        提示:
            本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。
        """
        return self.__topology_model.get_shared_synapse_state_position_iterator()

    def get_instantiated_synapse_state_shape(self) -> Sequence[int]:
        """获取突触状态张量的形状。

        返回值:
            突触状态张量的形状。

        提示:
            本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。
        """
        return self.__topology_model.get_instantiated_synapse_state_shape()

    def get_instantiated_synapse_state_position(self, presynaptic_position: np.ndarray,
                                                postsynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """获取给定突触的状态值在连接组突触状态张量中的位置。

        参数:
            presynaptic_position: 二维整数矩阵， 每行表示一个突触前神经元在其神经元组中的位置。
            postsynaptic_position: 二维整数矩阵， 每行表示一个突触后神经元在其神经元组中的位置。

        返回值:
            本方法的返回值依次为 `synapse_exists` 和 `synapse_state_position` 。

            `synapse_exists` 为向量，长度与 `presynaptic_position` 、 `postsynaptic_position` 和 `synapse_state_position` 的行数相等。

            `synapse_state_position` 为二维矩阵。 `presynaptic_position` 和 `postsynaptic_position` 对应位置上的行，
            即， `presynaptic_position[i]` 和 `postsynaptic_position[i]` 共同表示一个突触。
            当且仅当该突触存在， `synapse_exists[i]` 为真，该突触的状态值在状态张量中的位置为 `synapse_state_position[i]` 。

        提示:
            1. 本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

            2. 本方法假设连接组的突触状态张量不是标量。
        """
        synapse_state_position = self.__topology_model.get_instantiated_synapse_state_position(
            presynaptic_position, postsynaptic_position)
        synapse_exists = (synapse_state_position[:, 0] >= 0)
        return synapse_exists, synapse_state_position

    def get_instantiated_synapse_state_position_iterator(self) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """获取突触状态值位置迭代器。

        返回值:
            每次迭代一个突触，返回值依次包括：

            1. 突触前神经元在其神经元组中的位置；

            2. 突触后神经元在其神经元组中的位置；

            3. 突触状态值在突触状态张量中的位置。

        提示:
            本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。
        """
        return self.__topology_model.get_instantiated_synapse_state_position_iterator()

    def get_initial_synapse_state_value_by_synapse(self, presynaptic_position: np.ndarray, postsynaptic_position: np.ndarray,
                                                   synapse_state_name: str) -> tuple[np.ndarray, np.ndarray]:
        """获取突触状态初值。

        参数:
            presynaptic_position: 二维整数矩阵， 每行表示一个突触前神经元在其神经元组中的位置。
            postsynaptic_position: 二维整数矩阵， 每行表示一个突触后神经元在其神经元组中的位置。
            synapse_state_name: 突触状态名。

        返回值:
            本方法的返回值依次为 `synapse_exists` 和 `initial_synapse_state_value` 。

            `synapse_exists` 和 `initial_synapse_state_value` 均为向量，
            长度与 `presynaptic_position` 和 `postsynaptic_position` 的行数相等。

            `presynaptic_position` 和 `postsynaptic_position` 对应位置上的行，
            即， `presynaptic_position[i]` 和 `postsynaptic_position[i]` 共同表示一个突触。
            当且仅当该突触存在， `synapse_exists[i]` 为真。该突触的状态初值为 `initial_synapse_state_value[i]` 。

        抛出:
            KeyError: `synapse_state_name` 不是突触模型的状态名称。

        提示:
            本方法假设连接组的突触状态张量不是标量。
        """
        if not self.has_initial_synapse_state_value_tensor(synapse_state_name):
            raise KeyError(f"Synapse state name '{synapse_state_name}' is invalid.")

        synapse_state_position = self.__topology_model.get_shared_synapse_state_position(
            presynaptic_position, postsynaptic_position)
        synapse_exists = (synapse_state_position[:, 0] >= 0)

        dtype = self.__initial_synapse_state_value[synapse_state_name][0].dtype
        initial_synapse_state_value = np.empty((presynaptic_position.shape[0],), dtype=dtype)
        indices = tuple(synapse_state_position[synapse_exists].T)
        initial_synapse_state_value[synapse_exists] = self.__initial_synapse_state_value[synapse_state_name][0][indices]

        return synapse_exists, initial_synapse_state_value

    def get_initial_synapse_state_value_iterator_by_synapse(self, synapse_state_name: str) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], Any]]:
        """获取突触状态初值迭代器。

        参数:
            synapse_state_name: 突触状态名。

        返回值:
            每次迭代一个突触，返回值依次包括：

            1. 突触前神经元在其神经元组中的位置；

            2. 突触后神经元在其神经元组中的位置；

            3. 突触状态初值。

        抛出:
            KeyError: `synapse_state_name` 不是突触模型的状态名称。
        """
        if not self.has_initial_synapse_state_value_tensor(synapse_state_name):
            raise KeyError(f"Synapse state name '{synapse_state_name}' is invalid.")
        value = self.__initial_synapse_state_value[synapse_state_name][0]
        for pre_neuron_pos, post_neuron_pos, synapse_state_pos in self.get_shared_synapse_state_position_iterator():
            yield pre_neuron_pos, post_neuron_pos, value[synapse_state_pos]

    def get_delay(self) -> Union[float, Decimal]:
        """获取各个连接共同的延时。

        返回值:
            各个连接共同的延时。
        """
        return self.__delay
