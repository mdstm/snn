from collections.abc import Mapping, Iterator
from typing import Any, Union

from ..encoder_model import EncoderModel
from ..type.quantized_numeric_type import QuantizedNumericType


class Poisson(EncoderModel):
    __PARAMETERS = {
        "spike_rate_per_unit": float,   # 单位输入值对应的脉冲发放频率，正实数
        "time_step": float,             # 时间步长度，正实数
    }

    __STATES = {}

    def __init__(self, parameter: Mapping[str, Any]):
        """
        初始化Poisson实例。

        Args:
            parameter (Mapping[str, Any]): 包含神经元参数的字典。

        Raises:
            ValueError: 如果参数的值无效时抛出。
            TypeError: 如果参数或初始状态的类型无效时抛出。
        """
        parameter = dict(parameter)

        missing_parameters = [str(param) for param in self.__PARAMETERS.keys() if str(param) not in parameter]
        if missing_parameters:
            raise ValueError(f"Missing required parameters: {', '.join(missing_parameters)}")

        parameter = self.__check_and_convert(parameter, self.__PARAMETERS)
        super().__init__(parameter)

        if parameter["spike_rate_per_unit"] <= 0:
            raise ValueError("Parameter 'spike_rate_per_unit' must be a positive float.")
        if parameter["time_step"] <= 0:
            raise ValueError("Parameter 'time_step' must be a non-negative float.")

    def __check_and_convert(self, source: Mapping[str, Any], type_map: Mapping[str, Union[type, QuantizedNumericType]]) -> Mapping[str, Any]:
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
            value = source.get(name)
            original_value = value
            converted_value = expected_type(value)
            if original_value != converted_value:
                raise TypeError(
                    f"Value '{original_value}' for parameter/state '{name}' does not match the expected type '{expected_type}' after conversion.")
            result[name] = converted_value
        return result

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。

        Returns:
            str: 模型的唯一标识符 "Poisson"。
        """
        return "Poisson"

    def has_parameter(self, name: str) -> bool:
        """
        检查给定名称是否为有效的参数。

        Args:
            name (str): 要检查的参数名称。

        Returns:
            bool: 如果参数名称有效，则返回 True；否则返回 False。
        """
        name = str(name)
        return name in Poisson.__PARAMETERS

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
        if name in Poisson.__PARAMETERS:
            return Poisson.__PARAMETERS[name]
        raise KeyError(f"Parameter name '{name}' is not valid.")

    def get_parameter_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取参数类型的迭代器。

        Returns:
            Iterator[tuple[str, Union[type, QuantizedNumericType]]]: 参数名称及其类型的迭代器。
        """
        return iter(Poisson.__PARAMETERS.items())

    def __hash__(self) -> int:
        """
        获取对象的哈希值。

        Returns:
            int: 对象的哈希值。
        """
        hash_value = 0
        for name, parameter_type in self.get_parameter_type_iterator():
            hash_value ^= hash(name) ^ hash(parameter_type) ^ hash(self.get_converted_parameter(name))
        return hash_value

    def __eq__(self, other: 'Poisson') -> bool:
        """
        检查当前对象是否与另一个对象相等。

        Args:
            other (Poisson): 要比较的另一个对象。

        Returns:
            bool: 如果两个对象相等，则返回 True；否则返回 False。
        """
        if not isinstance(other, Poisson):
            return False
        for name, _ in self.get_parameter_type_iterator():
            if self.get_original_parameter(name) != other.get_converted_parameter(name):
                return False
        return True
