from typing import Any, Union
from collections.abc import Mapping, Iterator

import numpy as np

from ..neuron_model import NeuronModel
from ..type.quantized_numeric_type import QuantizedNumericType


class LIF(NeuronModel):
    __PARAMETERS = {
        "capacitance": float,                                                                                         # 电容，正实数
        "resistance": float,                                                                                          # 电阻，正实数
        "time_step": float,                                                                                           # 时间步长度，非负实数
        "voltage_rest": QuantizedNumericType.get("int", bit_width=16, signed=True),                 # 静息电位，16位有符号整数
        "threshold": QuantizedNumericType.get("int", bit_width=16, signed=True),                    # 发放阈值，16位有符号整数
        "voltage_reset_value": QuantizedNumericType.get("int", bit_width=16, signed=True),          # 重置电位，16位有符号整数
        "voltage_reset_decrement": QuantizedNumericType.get("int", bit_width=16, signed=False),     # 重置电位下降，16位无符号整数
        "refractory_period": float,  # 不应期，正实数或无穷大
        "voltage_initial": QuantizedNumericType.get("int", bit_width=16, signed=True),              # 初始膜电位，16位有符号整数
    }

    __STATES = {
        "weight_sum": QuantizedNumericType.get("int", bit_width=16, signed=True),                   # 权重和，16位有符号整数，默认0
        "voltage": QuantizedNumericType.get("int", bit_width=16, signed=True),                      # 膜电位，16位有符号整数，默认 voltage_rest
    }

    def __init__(self, parameter: Mapping[str, Any], initial_state: Mapping[str, Any]):
        """
        初始化LIF实例。

        Args:
            parameter (Mapping[str, Any]): 包含神经元参数的字典。
            initial_state (Mapping[str, Any]): 包含神经元初始状态的字典。

        Raises:
            ValueError: 如果 'voltage_reset_value' 和 'voltage_reset_decrement' 均未提供或均提供时抛出。
            ValueError: 如果参数 'capacitance'、'resistance'、'time_step' 或 'refractory_period' 的值无效时抛出。
            TypeError: 如果参数或初始状态的类型无效时抛出。
        """
        parameter = dict(parameter)
        initial_state = dict(initial_state)

        if ('voltage_reset_value' in parameter) == ('voltage_reset_decrement' in parameter):
            raise ValueError(
                "One of 'voltage_reset_value' or 'voltage_reset_decrement' must be provided, and they cannot be both provided.")

        missing_parameters = [
            param for param in self.__PARAMETERS.keys()
            if param not in parameter and param not in ['voltage_reset_value', 'voltage_reset_decrement']
        ]
        if missing_parameters:
            raise ValueError(f"Missing required parameters: {', '.join(missing_parameters)}")

        parameter = self.__check_and_convert(parameter, self.__PARAMETERS)
        initial_state = self.__check_and_convert(initial_state, self.__STATES)

        super().__init__(parameter, initial_state)

        if parameter["capacitance"] <= 0:
            raise ValueError("Parameter 'capacitance' must be a positive float.")
        if parameter["resistance"] <= 0:
            raise ValueError("Parameter 'resistance' must be a positive float.")
        if parameter["time_step"] < 0:
            raise ValueError("Parameter 'time_step' must be a non-negative float.")
        if not (parameter["refractory_period"] > 0 or parameter["refractory_period"] == float("inf")):
            raise ValueError("Parameter 'refractory_period' must be a positive float or infinity.")

        self.__set_default_states()

    def __check_and_convert(self, source: Mapping[str, Any], type_map: Mapping[str, Union[type, QuantizedNumericType]]):
        """
        检查和转换源字典中的值为指定类型。

        Args:
            source (Mapping[str, Any]): 源字典。
            type_map (Mapping[str, Union[type, QuantizedNumericType]]): 类型映射字典。

        Raises:
            TypeError: 如果值的类型无效。
            ValueError: 如果值的规则无效。
        """
        result = {}
        for name, expected_type in type_map.items():
            name = str(name)
            original_value = source.get(name)
            if original_value is None:
                continue
            converted_value = expected_type(original_value)
            if original_value != converted_value:
                raise TypeError(
                    f"Value '{original_value}' for parameter/state '{name}' does not match the expected type '{expected_type}' after conversion.")
            result[name] = converted_value
        return result

    def __set_default_states(self):
        """
        设置默认状态值。
        """
        if self.get_original_initial_state_value("weight_sum") is None:
            self.set_initial_state_value("weight_sum", 0)

        if self.get_original_initial_state_value("voltage") is None:
            voltage_rest = self.get_original_parameter("voltage_rest")
            self.set_initial_state_value("voltage", voltage_rest)

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。

        Returns:
            str: 模型的唯一标识符 "LIF"。
        """
        return "LIF"

    def has_parameter(self, name: str) -> bool:
        """
        检查给定名称是否为有效的参数。

        Args:
            name (str): 要检查的参数名称。

        Returns:
            bool: 如果参数名称有效，则返回 True；否则返回 False。
        """
        name = str(name)
        return name in LIF.__PARAMETERS

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
        if name in LIF.__PARAMETERS:
            return LIF.__PARAMETERS[name]
        raise KeyError(f"Parameter name '{name}' is not valid.")

    def get_parameter_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取参数类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 参数名称及其类型的迭代器。
        """
        return iter(LIF.__PARAMETERS.items())

    def has_state(self, name: str) -> bool:
        """
        检查给定名称是否为有效的状态。

        Args:
            name (str): 要检查的状态名称。

        Returns:
            bool: 如果状态名称有效，则返回 True；否则返回 False。
        """
        name = str(name)
        return name in LIF.__STATES

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
        if name in LIF.__STATES:
            return LIF.__STATES[name]
        raise KeyError(f"State name '{name}' is not valid.")

    def get_state_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取状态类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 状态名称及其类型的迭代器。
        """
        return iter(LIF.__STATES.items())

    def get_input_state(self) -> str:
        """
        获取输入状态的名称。

        Returns:
            str: 输入状态的名称 "weight_sum"。
        """
        return "weight_sum"

    def __hash__(self) -> int:
        """
        获取对象的哈希值。

        Returns:
            int: 对象的哈希值。
        """
        hash_value = 0
        for name, parameter_type in self.get_parameter_type_iterator():
            val = self.get_converted_parameter(name)                    # 新建
            if isinstance(val, np.ndarray):                             # 新建
                val = val.tobytes()                                     # 新建
            hash_value ^= hash(name) ^ hash(parameter_type) ^ hash(val) # 修改
        for name, state_type in self.get_state_type_iterator():
            val = self.get_converted_initial_state_value(name)      # 新建
            if isinstance(val, np.ndarray):                         # 新建
                val = val.tobytes()                                 # 新建
            hash_value ^= hash(name) ^ hash(state_type) ^ hash(val) # 修改
        return hash_value

    def __eq__(self, other: 'LIF') -> bool:
        """
        检查当前对象是否与另一个对象相等。

        Args:
            other (LIF): 要比较的另一个对象。

        Returns:
            bool: 如果两个对象相等，则返回 True；否则返回 False。
        """
        if not isinstance(other, LIF):
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
