from abc import ABC


class Element(ABC):
    """SNN中所有元素的公共基类。"""
    def __init__(self):
        self.__element_id: str = None

    def set_element_id(self, element_id: str):
        """设置元素标识符。

        参数:
            element_id: 元素标识符。
        """
        self.__element_id = str(element_id)

    def get_element_id(self) -> str:
        """获取元素标识符。

        返回值:
            元素标识符。
        """
        return self.__element_id
