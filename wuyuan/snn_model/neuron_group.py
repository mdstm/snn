import numpy as np
from collections.abc import Iterator, Mapping, Sequence
from .node import Node
from .element_stateful import ElementStateful
from .neuron_model import NeuronModel


class NeuronGroup(Node, ElementStateful):
    def __init__(self, neuron_model: NeuronModel, shape: Sequence[int],
                 initial_state_value: Mapping[str, tuple[np.ndarray, bool]] = None):
        """
        初始化 NeuronGroup 实例。

        Args:
            neuron_model (NeuronModel): 神经元模型实例。
            shape (Sequence[int]): 神经元组的形状。
            initial_state_value (Mapping[str, tuple[np.ndarray, bool]], optional): 初始状态值的映射字典，默认值为 None。

        Raises:
            TypeError: 如果 neuron_model 不是 NeuronModel 实例。
        """
        if not isinstance(neuron_model, NeuronModel):
            raise TypeError("neuron_model must be an instance of NeuronModel.")

        super().__init__(shape)
        self.__neuron_model: NeuronModel = neuron_model
        self.__initial_neuron_state_value: dict[str, tuple[np.ndarray, bool]] = {}

        if initial_state_value:
            for state_name, (state_value, constant) in initial_state_value.items():
                self.set_initial_state_value_tensor(state_name, state_value, constant)

    def get_neuron_model(self) -> NeuronModel:
        """
        获取神经元模型。

        Returns:
            NeuronModel: 神经元模型实例。
        """
        return self.__neuron_model

    def set_initial_state_value_tensor(self, state_name: str, state_value: np.ndarray, constant: bool = False):
        """
        设置初始状态值张量。

        Args:
            state_name (str): 状态名称。
            state_value (np.ndarray): 状态值张量。
            constant (bool): 是否为常量。

        Raises:
            KeyError: 如果状态名称在神经元模型中无效。
            ValueError: 如果状态值张量的形状与神经元组的形状不匹配。
            TypeError: 如果 state_value 不是 NumPy 数组或 constant 不是 bool 类型。
        """
        state_value = np.asarray(state_value)
        constant = bool(constant)

        # if not hasattr(self, '__neuron_model') or not hasattr(self, '__initial_neuron_state_value'): # 修改
        #     raise RuntimeError("NeuronGroup instance is not fully initialized.")                     # 修改

        state_name = str(state_name)
        if not self.__neuron_model.has_state(state_name):
            raise KeyError(f"State name '{state_name}' is not valid.")

        state_value = self.__neuron_model.check_state_value(state_name, state_value)

        if state_value.shape != self.get_shape():
            raise ValueError(
                f"Shape of state_value {state_value.shape} does not match NeuronGroup shape {self.get_shape()}.")
        self.__initial_neuron_state_value[state_name] = (state_value, constant)

    def has_initial_state_value_tensor(self, state_name: str) -> bool:
        """
        检查是否具有指定状态名称的初始状态值张量。

        Args:
            state_name (str): 状态名称。

        Returns:
            bool: 如果存在初始状态值张量，则返回 True，否则返回 False。
        """
        state_name = str(state_name)
        return state_name in self.__initial_neuron_state_value

    def get_initial_state_value_tensor(self, state_name: str) -> np.ndarray:
        """
        获取指定状态名称的初始状态值张量。

        Args:
            state_name (str): 状态名称。

        Returns:
            np.ndarray: 初始状态值张量。

        Raises:
            KeyError: 如果状态名称在神经元模型中无效或对应的初始状态值未设置。
        """
        state_name = str(state_name)

        if not self.__neuron_model.has_state(state_name):
            raise KeyError(f"State name '{state_name}' is not valid.")

        if not self.has_initial_state_value_tensor(state_name):
            raise KeyError(f"Initial state value for state name '{state_name}' is not set.")

        return self.__initial_neuron_state_value[state_name][0]

    def get_initial_state_value_tensor_iterator(self) -> Iterator[tuple[str, np.ndarray]]:
        """
        获取初始状态值张量的迭代器。

        Returns:
            Iterator[Tuple[str, np.ndarray]]: 一个迭代器，每次迭代返回一个包含状态名称和对应初始状态值张量的元组。
        """
        for state_name, (state_value, constant) in self.__initial_neuron_state_value.items():
            yield state_name, state_value

    def is_state_constant(self, state_name: str) -> bool:
        """
        判断给定神经元状态是否常量。

        Args:
            state_name (str): 状态名称。

        Returns:
            bool: 如果状态是常量，则返回 True，否则返回 False。

        Raises:
            KeyError: 如果状态名称在神经元模型中无效或对应的初始状态值未设置。
        """
        state_name = str(state_name)

        if not self.__neuron_model.has_state(state_name):
            raise KeyError(f"State name '{state_name}' is not valid.")

        if not self.has_initial_state_value_tensor(state_name):
            raise KeyError(f"Initial state value for state name '{state_name}' is not set.")

        return self.__initial_neuron_state_value[state_name][1]