from abc import ABC, abstractmethod

from .element import Element


class ElementModel(Element):
    """模型元素。"""
    @staticmethod
    @abstractmethod
    def get_model_id() -> str:
        """获取模型标识符。

        返回值:
            模型标识符。
        """
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """计算哈希值。

        返回值:
            哈希值。
        """
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other: "ElementModel") -> bool:
        """判断本模型元素与给定模型元素是否相等。

        参数:
            other: 模型元素。

        返回值:
            本模型元素与给定模型元素是否相等。
        """
        raise NotImplementedError()
