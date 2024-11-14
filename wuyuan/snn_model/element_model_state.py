from abc import ABC, abstractmethod
from typing import Any, Union
from collections.abc import Iterator

import numpy as np

from .type.quantized_numeric_type import QuantizedNumericType

class ElementModelState(ABC):
    def __init__(self):
        """
        初始化 ElementModelState 实例。
        """
        self.__state: dict[str, Any] = {}

    @abstractmethod
    def has_state(self, name: str) -> bool:
        """
        检查状态是否有效。

        Args:
            name (str): 状态名称。

        Returns:
            bool: 状态名称是否有效。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state_type(self, name: str) -> Union[type, QuantizedNumericType]:
        """
        获取状态类型。

        Args:
            name (str): 状态名称。

        Raises:
            KeyError: 如果 name 不是有效的状态名称。

        Returns:
            Union[type, QuantizedNumericType]: 状态的类型。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取状态类型迭代器。

        Returns:
            Iterator[Tuple[str, Union[type, QuantizedNumericType]]]: 每次迭代返回一个状态的名称和类型。
        """
        raise NotImplementedError()

    @abstractmethod
    def check_state_value(self, name: str, value: np.ndarray) -> np.ndarray:
        """
        检查各个状态值是否均满足模型元素的要求。
        若 value 中存在不满足模型元素要求的状态值，则本方法抛出ValueError。

        Args:
            name (str): 状态名称。
            value (np.ndarray): 状态值。

        Raises:
            ValueError: 如果状态值无效。

        Returns:
            np.ndarray: 转换后的状态值。
        """
        raise NotImplementedError()

    def set_initial_state_value(self, name: str, value: Any):
        """
        设置状态的初始值。

        Args:
            name (str): 状态名称。
            value (Any): 状态初始值。

        Raises:
            KeyError: 如果 name 不是有效的状态名称。
        """
        name = str(name)
        if not self.has_state(name):
            raise KeyError(f"State name '{name}' is not valid.")
        self.__state[name] = value

    def get_original_initial_state_value(self, name: str) -> Any:
        """
        获取未经转换的状态初始值。

        Args:
            name (str): 状态名称。

        Raises:
            KeyError: 如果 name 不是有效的状态名称。

        Returns:
            Any: 未经转换的状态初始值，如果 name 对应的状态值未设置，则返回 None。
        """
        name = str(name)
        if not self.has_state(name):
            raise KeyError(f"State name '{name}' is not valid.")
        return self.__state.get(name)

    def get_converted_initial_state_value(self, name: str) -> Any:
        """
        获取已被转换为状态类型的状态初始值。

        Args:
            name (str): 状态名称。

        Raises:
            KeyError: 如果 name 不是有效的状态名称。

        Returns:
            Any: 已转换的状态初始值，如果 name 对应的状态值未设置，则返回 None。
        """
        name = str(name)
        value = self.get_original_initial_state_value(name)
        if value is None:
            return None
        state_type = self.get_state_type(name)
        return state_type(value)

    def get_state_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType], Any]]:
        """
        获取状态迭代器。

        Returns:
            Iterator[Tuple[str, Union[type, QuantizedNumericType], Any]]: 每次迭代返回一个状态的名称、类型和未经转换的初值。
        """
        for name, state_type in self.get_state_type_iterator():
            value = self.__state.get(name)
            yield (name, state_type, value)
