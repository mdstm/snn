from collections.abc import Mapping, Iterator
from typing import Any, Union

from ..decoder_model import DecoderModel
from ..type.quantized_numeric_type import QuantizedNumericType


class SpikeCount(DecoderModel):
    def __init__(self, parameter: Mapping[str, Any]):
        """
        初始化SpikeCount实例。

        Args:
            parameter (Mapping[str, Any]): 包含神经元参数的字典。

        Raises:
            ValueError: 如果参数的值无效时抛出。
            TypeError: 如果参数或初始状态的类型无效时抛出。
        """
        super().__init__({})

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。

        Returns:
            str: 模型的唯一标识符 "SpikeCount"。
        """
        return "SpikeCount"

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

    def __eq__(self, other: 'SpikeCount') -> bool:
        """
        检查当前对象是否与另一个对象相等。

        Args:
            other (SpikeCount): 要比较的另一个对象。

        Returns:
            bool: 如果两个对象相等，则返回 True；否则返回 False。
        """
        if not isinstance(other, SpikeCount):
            return False
        for name, _ in self.get_parameter_type_iterator():
            if self.get_original_parameter(name) != other.get_converted_parameter(name):
                return False
        return True
