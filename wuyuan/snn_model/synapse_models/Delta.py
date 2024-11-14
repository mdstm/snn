from typing import Any, Union
from collections.abc import Iterator, Mapping

import numpy as np

from ..synapse_model import SynapseModel
from ..type.quantized_numeric_type import QuantizedNumericType


class Delta(SynapseModel):
    def __init__(self, parameter: Mapping[str, Any], initial_state: Mapping[str, Any]):
        """
        初始化 Delta 类的实例。

        Args:
            parameter (Mapping[str, Any]): 包含模型参数的字典。
            initial_state (Mapping[str, Any]): 包含模型初始状态的字典。

        Raises:
            ValueError: 如果未提供 bit_width_weight、weight 或 current 的初始值，则抛出此异常。
        """
        parameter = dict(parameter)
        initial_state = dict(initial_state)

        if 'bit_width_weight' not in parameter:
            raise ValueError("Parameter 'bit_width_weight' must be provided.")

        if 'weight' not in initial_state:
            raise ValueError("Initial state 'weight' must be provided.")
        if 'current' not in initial_state:
            initial_state['current'] = 0

        self.__PARAMETERS = {}
        self.__STATES = {}

        original_bit_width_weight = parameter['bit_width_weight']
        converted_bit_width_weight = int(original_bit_width_weight)
        if original_bit_width_weight != converted_bit_width_weight:
            raise TypeError(
                f"Value '{original_bit_width_weight}' for parameter/state 'bit_width_weight' does not match the expected type 'int' after conversion.")
        if converted_bit_width_weight not in [2, 4, 6, 8, 16]:
            raise ValueError(
                f"Value '{converted_bit_width_weight}' for parameter/state 'bit_width_weight' is not in the valid range [2, 4, 6, 8, 16].")
        parameter['bit_width_weight'] = converted_bit_width_weight
        self.__PARAMETERS['bit_width_weight'] = int

        type = QuantizedNumericType.get("int", bit_width=converted_bit_width_weight, signed=True)
        initial_state["weight"] = self.__check_and_convert("weight", initial_state['weight'], type)
        self.__STATES["weight"] = type

        type = QuantizedNumericType.get("int", bit_width=16, signed=True)
        initial_state["current"] = self.__check_and_convert("current", initial_state['current'], type)
        self.__STATES["current"] = type

        super().__init__(parameter, initial_state)

    def __check_and_convert(self, name: str, value: Any, type: Union[type, QuantizedNumericType]):
        """
        检查并转换参数或状态的值。

        Args:
            name (str): 参数或状态的名称。
            value (Any): 参数或状态的原始值。
            type (Union[type, QuantizedNumericType]): 用于转换的类型或量化数值类型。

        Raises:
            ValueError: 如果在类型转换后值无效，则抛出此异常。

        Returns:
            Any: 转换后的值。
        """
        original_value = value
        converted_value = type(value)
        if original_value != converted_value:
            raise ValueError(
                f"Value '{original_value}' for parameter/state '{name}' does not match the expected type '{type}' after conversion.")
        return converted_value

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。

        Returns:
            str: 模型的唯一标识符 "Delta"。
        """
        return "Delta"

    def has_parameter(self, name: str) -> bool:
        """
        检查给定名称的参数是否存在于模型中。

        Args:
            name (str): 要检查的参数名称。

        Returns:
            bool: 如果参数存在，则返回True；否则返回False。
        """
        name = str(name)
        return name in self.__PARAMETERS

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
        if name in self.__PARAMETERS:
            return self.__PARAMETERS[name]
        raise KeyError(f"Parameter name '{name}' is not valid.")

    def get_parameter_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取参数类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 参数名称及其类型的迭代器。
        """
        return iter(self.__PARAMETERS.items())

    def has_state(self, name: str) -> bool:
        """
        检查给定名称是否为有效的状态。

        Args:
            name (str): 要检查的状态名称。

        Returns:
            bool: 如果状态名称有效，则返回 True；否则返回 False。
        """
        name = str(name)
        return name in self.__STATES

    def get_state_type(self, name: str) -> Union[type, QuantizedNumericType]:
        """
        获取指定状态的类型。

        Args:
            name (str): 状态名称。

        Raises:
                KeyError: 如果 name 不是有效的状态名称。

        Returns:
                Union[type, QuantizedNumericType]: 状态的类型。
        """
        name = str(name)
        if name in self.__STATES:
            return self.__STATES[name]
        raise KeyError(f"State name '{name}' is not valid.")

    def get_state_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取状态类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 状态名称及其类型的迭代器。
        """

        return iter(self.__STATES.items())

    def get_weight_state(self) -> str:
        """
        获取权重状态的名称。

        Returns:
            str: 权重状态的名称 "weight"。
        """
        return "weight"

    def get_output_state(self) -> str:
        """
        获取输出状态的名称。

        Returns:
            str: 输出状态的名称 "current"。
        """
        return "current"

    def __hash__(self) -> int:
        """
        获取对象的哈希值。

        Returns:
            int: 对象的哈希值。
        """
        hash_value = 0
        for name, parameter_type in self.get_parameter_type_iterator():
            hash_value ^= hash(name) ^ hash(parameter_type) ^ hash(self.get_converted_parameter(name))
        for name, state_type in self.get_state_type_iterator():
            val = self.get_converted_initial_state_value(name)      # 新建
            if isinstance(val, np.ndarray):                         # 新建
                val = val.tobytes()                                 # 新建
            hash_value ^= hash(name) ^ hash(state_type) ^ hash(val) # 修改
        return hash_value

    def __eq__(self, other: 'Delta') -> bool:
        """
        检查当前对象是否与另一个对象相等。

        Args:
            other (Delta): 要比较的另一个对象。

        Returns:
            bool: 如果两个对象相等，则返回 True；否则返回 False。
        """
        if not isinstance(other, Delta):
            return False
        for name, _ in self.get_parameter_type_iterator():
            if self.get_converted_parameter(name) != other.get_converted_parameter(name):
                return False
        for name, _ in self.get_state_type_iterator():
            if self.get_converted_initial_state_value(name) != other.get_converted_initial_state_value(name):
                return False
        return True

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
        name = str(name)
        value = np.asarray(value)
        expected_type = self.get_state_type(name)
        converted_value = expected_type(value)
        if not np.all(value == converted_value):
            raise ValueError(
                f"State value for '{name}' cannot be converted to the expected type {expected_type}.")
        return converted_value
