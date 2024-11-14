from collections.abc import Callable


class IDUniquer:
    """唯一标识符生成器。

    参数:
        exists: 用于判断给定标识符是否已存在的函数。
        prefix: 标识符前缀。若 `prefix` 为 ``None`` ，则标识符前缀为空字符串。
    """
    def __init__(self, exists: Callable[[str], bool], prefix: str = None):
        if not isinstance(exists, Callable):
            raise TypeError(
                "exists must be a callable that takes a str as input and "
                "returns a bool.")
        self.__exists = exists
        self.__prefix: str = "" if prefix is None else str(prefix)
        self.__count: int = 0

    def get(self, suffix: str = None) -> str:
        """获取新的唯一标识符。

        参数:
            suffix: 标识符后缀。若 `suffix` 为 ``None`` ，则标识符后缀为空字符串。

        返回值:
            新的唯一标识符。
        """
        suffix_ = "" if suffix is None else str(suffix)
        id_ = self.__prefix + "_" + str(self.__count) + "_" + suffix_
        self.__count += 1
        while self.__exists(id_):
            id_ = self.__prefix + "_" + str(self.__count) + "_" + suffix_
            self.__count += 1
        return id_
