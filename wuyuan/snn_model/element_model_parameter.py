from abc import ABC, abstractmethod
from typing import Any, Union
from collections.abc import Iterator
from .type.quantized_numeric_type import QuantizedNumericType

class ElementModelParameter(ABC):
    def __init__(self):
        """
        初始化 ElementModelParameter 实例。
        """
        self.__parameter: dict[str, Any] = {}

    @abstractmethod
    def has_parameter(self, name: str) -> bool:
        """
        检查参数是否有效。

        Args:
            name (str): 参数名称。

        Returns:
            bool: 参数名称是否有效。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_parameter_type(self, name: str) -> Union[type, QuantizedNumericType]:
        """
        获取参数类型。

        Args:
            name (str): 参数名称。

        Raises:
            KeyError: 如果 name 不是有效的参数名称。

        Returns:
            Union[type, QuantizedNumericType]: 参数的类型。
        """
        raise NotImplementedError()

    @abstractmethod
    def get_parameter_type_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType]]]:
        """
        获取参数类型迭代器。

        Returns:
            Iterator[Tuple[str, Union[type, QuantizedNumericType]]]: 每次迭代返回一个参数的名称和类型。
        """
        raise NotImplementedError()

    def set_parameter(self, name: str, value: Any):
        """
        设置参数的值。

        Args:
            name (str): 参数名称。
            value (Any): 参数值。

        Raises:
            KeyError: 如果 name 不是有效的参数名称。
        """
        name = str(name)
        if not self.has_parameter(name):
            raise KeyError(f"Parameter name '{name}' is not valid.")
        self.__parameter[name] = value

    def get_original_parameter(self, name: str) -> Any:
        """
        获取未经转换的参数值。

        Args:
            name (str): 参数名称。

        Raises:
            KeyError: 如果 name 不是有效的参数名称。

        Returns:
            Any: 未经转换的参数值，如果 name 对应的参数值未设置，则返回 None。
        """
        name = str(name)
        if not self.has_parameter(name):
            raise KeyError(f"Parameter name '{name}' is not valid.")
        return self.__parameter.get(name)

    def get_converted_parameter(self, name: str) -> Any:
        """
        获取已被转换为参数类型的参数值。

        Args:
            name (str): 参数名称。

        Raises:
            KeyError: 如果 name 不是有效的参数名称。

        Returns:
            Any: 已转换的参数值，如果 name 对应的参数值未设置，则返回 None。
        """
        name = str(name)
        value = self.get_original_parameter(name)
        if value is None:
            return None
        param_type = self.get_parameter_type(name)
        return param_type(value)

    def get_parameter_iterator(self) -> Iterator[tuple[str, Union[type, QuantizedNumericType], Any]]:
        """
        获取参数迭代器。

        Returns:
            Iterator[Tuple[str, Union[type, QuantizedNumericType], Any]]: 每次迭代返回一个参数的名称、类型和未经转换的值。
        """
        for name, param_type in self.get_parameter_type_iterator():
            value = self.__parameter.get(name)
            yield (name, param_type, value)
