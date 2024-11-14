from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np


class QuantizedNumericType(ABC):
    """经过量化的数值类型。"""
    @staticmethod
    def register(numeric_type_class: type):
        """注册具体的数值类型类。

        参数:
            numeric_type_class:
                具体的数值类型类。 `numeric_type_class` 必须是
                QuantizedNumericType类的派生类。
        """
        if not (
            isinstance(numeric_type_class, type) and
                issubclass(numeric_type_class, QuantizedNumericType)):
            raise TypeError(
                "numeric_type_class must be a derived class of "
                "QuantizedNumericType.")
        numeric_type_class_id = str(
            numeric_type_class.get_numeric_type_class_id())
        if numeric_type_class_id in QuantizedNumericType.__numeric_type_class:
            raise TypeError(
                f"A numeric type class with the same ID "
                f"{numeric_type_class_id} has already been registered.")
        QuantizedNumericType.__numeric_type_class[numeric_type_class_id] = \
            numeric_type_class

    @staticmethod
    def get(numeric_type_class_id: str, **kwargs) -> "QuantizedNumericType":
        """获取经过量化的数值类型。

        参数:
            numeric_type_class_id: 经过量化的数值类型的类标识符。
            \*\*kwargs: 用于构造经过量化的数值类型的实例的参数。

        返回值:
            经过量化的数值类型的实例。
        """
        quantized_numeric_type = QuantizedNumericType.__numeric_type_class[
            str(numeric_type_class_id)](
                **kwargs)
        instance_set = QuantizedNumericType.__numeric_type_instance
        quantized_numeric_type_ = instance_set.get(
            quantized_numeric_type, None)
        if quantized_numeric_type_ is not None:
            return quantized_numeric_type_
        instance_set[quantized_numeric_type] = quantized_numeric_type
        return quantized_numeric_type

    @staticmethod
    @abstractmethod
    def get_numeric_type_class_id() -> str:
        """获取经过量化的数值类型的类标识符。

        返回值:
            经过量化的数值类型的类标识符。
        """
        raise NotImplementedError()

    def __init__(self):
        self.__parameter: dict[str, Any] = dict()

    def get_parameter(self) -> Mapping[str, Any]:
        """获取从参数名称到参数值的映射关系。

        返回值:
            从参数名称到参数值的映射关系。
        """
        return self.__parameter

    @abstractmethod
    def quantize(self, value: np.ndarray) -> np.ndarray:
        """量化给定数值。

        参数:
            value: 数值。

        返回值:
            经过量化的数值。

        提示:
            本方法返回数组的每个元素是 `value` 对应元素的量化结果。
        """
        raise NotImplementedError()

    def __call__(self, value: np.ndarray) -> np.ndarray:
        """通过调用quantize方法量化给定数值。

        参数:
            value: 数值。

        返回值:
            经过量化的数值。
        """
        return self.quantize(value)

    @abstractmethod
    def __eq__(self, other: "QuantizedNumericType") -> bool:
        """判断本实例与给定实例是否相等。

        参数:
            other: 给定实例。

        返回值:
            本实例与给定实例是否相等。
        """
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """计算本实例的哈希值。

        返回值:
            本实例的哈希值。
        """
        raise NotImplementedError()

    def _add_parameter(self, name: str, value: Any):
        """添加参数。

        参数:
            name: 参数名称。
            value: 参数值。
        """
        name_ = str(name)
        if name_ in self.__parameter:
            raise ValueError(
                f"A parameter with the same name {name_} has already been "
                f"added.")
        self.__parameter[name_] = value

    __numeric_type_class: dict[str, type] = dict()
    __numeric_type_instance: dict[
        "QuantizedNumericType", "QuantizedNumericType"] = \
        dict()
