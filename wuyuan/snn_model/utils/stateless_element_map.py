from collections.abc import Iterator

from ..element import Element
from .id_uniquer import IDUniquer


def create_stateless_element_map_class(
        element_type: type, element_type_name_id_prefix: str = None,
        element_type_name_class: str = None,
        element_type_name_member: str = None,
        element_type_name_doc: str = None) \
        -> type:
    """创建从标识符到无状态元素的映射关系类。

    参数:
        element_type: 元素类型。
        element_type_name_id_prefix:
            作为标识符前缀使用的元素类型名称。若为 ``None`` ，则本函数使用
            ``element_type.name`` 。
        element_type_name_class:
            在映射关系类的名称中使用的元素类型名称。若为 ``None`` ，则本函数使用
            ``element_type.name`` 。
        element_type_name_member:
            在映射关系类成员的名称和参数列表中使用的元素类型名称。若为 ``None`` ，则本函数
            使用 ``element_type.name`` 。
        element_type_name_doc:
            在注释中使用的元素类型名称。若为 ``None`` ，则本函数使用
            ``element_type.name`` 。

    返回值:
        创建的从标识符到无状态元素的映射关系类。

    提示:
        1. 本函数创建一个类，该类实现从标识符到 `element_type` 实例的映射关系，其中各个
        `element_type` 实例不相等。

        2. `element_type` 必须是Element类的派生类。

        3. 本函数创建的类假设 `element_type` 实例可哈希（hashable），且不可变
        （immutable）。
    """
    if not issubclass(element_type, Element):
        raise TypeError("element_type must be a derived class of Element.")
    element_type_name_id_prefix = element_type.__name__ \
        if element_type_name_id_prefix is None \
        else str(element_type_name_id_prefix)
    if element_type_name_id_prefix == "":
        raise ValueError(
            "element_type_name_id_prefix must not be an empty string.")
    element_type_name_class = element_type.__name__ \
        if element_type_name_class is None \
        else str(element_type_name_class)
    if element_type_name_class == "":
        raise ValueError(
            "element_type_name_class must not be an empty string.")
    element_type_name_member = element_type.__name__ \
        if element_type_name_member is None \
        else str(element_type_name_member)
    if element_type_name_member == "":
        raise ValueError(
            "element_type_name_member must not be an empty string.")
    element_type_name_doc = element_type.__name__ \
        if element_type_name_doc is None \
        else str(element_type_name_doc)
    if element_type_name_doc == "":
        raise ValueError("element_type_name_doc must not be an empty string.")
    locals_ = locals()
    element_type_name = element_type.__name__
    exec(
        f"""
class {element_type_name_class}StatelessElementMap:
    \"\"\"从标识符到{element_type_name_doc}且各个{element_type_name_doc}互不相等的映
    射关系。

    提示:
        1. 本类实现从标识符到{element_type_name}实例的映射关系，其中各个
        {element_type_name}实例不相等。

        2. 本类假设{element_type_name}实例可哈希（hashable），且不可变（immutable）。
    \"\"\"
    def __init__(self):
        self.__{element_type_name_member}_id_to_element: \
            dict[str, element_type] = \
            dict()
        self.__{element_type_name_member}_element_to_id: \
            dict[element_type, str] = \
            dict()
        self.__{element_type_name_member}_id_uniquer: IDUniquer = IDUniquer(
            self.{element_type_name_member}_contains_id,
            element_type_name_id_prefix)
        self.__{element_type_name_member}_element_instance_id: set[int] = set()

    def {element_type_name_member}_get(
            self, id_: str,
            {element_type_name_member}: element_type = None) \
            -> tuple[str, element_type]:
        \"\"\"获取{element_type_name_doc}。

        参数:
            id_: {element_type_name_doc}标识符。
            {element_type_name_member}: {element_type_name_doc}。

        返回值:
            已被添加到映射关系的{element_type_name_doc}。

        提示:
            1. 本方法在各种情况下的行为如下：

            1.1. 在 `{element_type_name_member}` 为 ``None`` 的情况下，本方法在映射
            关系中查找 `id_` 的映射项，返回 `id_` 和对应的{element_type_name_doc}，而
            不更新映射关系。若映射关系不包含 `id_` ，则本方法抛出KeyError。

            1.2. 在 `{element_type_name_member}` 不为 ``None`` 的情况下，本方法判断
            `{element_type_name_member}` 与映射关系中现有的{element_type_name_doc}
            是否相等。

            1.2.1. 若 `{element_type_name_member}` 与现有{element_type_name_doc}
            相等，则本方法返回现有{element_type_name_doc}。

            1.2.2. 若 `{element_type_name_member}` 与现有{element_type_name_doc}
            不相等，则本方法将 `{element_type_name_member}` 的映射项添加到映射关系，并
            返回 `{element_type_name_member}` 。在这种情况下，若映射关系不包含
            `id_` ，则新映射项中的标识符为 `id_` ；否则，本方法生成一个不存在于映射关系中的
            标识符作为新映射项的标识符。

            2. 对于本方法返回的{element_type_name_doc}，本方法更新其标识符，使其与映射关
            系在本方法返回时的状态一致。
        \"\"\"
        id_ = str(id_)
        if {element_type_name_member} is None:
            return self.__{element_type_name_member}_id_to_element[id_]
        if not isinstance({element_type_name_member}, element_type):
            raise TypeError(
                \"{element_type_name_member} must be a {element_type_name} \"
                \"instance.\")
        id_existing = self.__{element_type_name_member}_element_to_id.get(
            {element_type_name_member}, None)
        if id_existing is not None:
            return self.__{element_type_name_member}_id_to_element[id_existing]
        if id_ not in self.__{element_type_name_member}_id_to_element:
            id_new = id_
        else:
            id_new = self.__{element_type_name_member}_id_uniquer.get(id_)
        try:
            self.__{element_type_name_member}_id_to_element[id_new] = \
                {element_type_name_member}
            self.__{element_type_name_member}_element_to_id[
                {element_type_name_member}] = \
                id_new
            self.__{element_type_name_member}_element_instance_id.add(
                id({element_type_name_member}))
        except:
            self.__{element_type_name_member}_id_to_element.pop(id_new, None)
            self.__{element_type_name_member}_element_to_id.pop(
                {element_type_name_member}, None)
            self.__{element_type_name_member}_element_instance_id.discard(
                id({element_type_name_member}))
            raise
        {element_type_name_member}.set_element_id(id_new)
        return {element_type_name_member}

    def {element_type_name_member}_remove(self, id_: str) -> element_type:
        \"\"\"移除给定标识符和对应的{element_type_name_doc}。

        参数:
            id_: {element_type_name_doc}标识符。

        返回值:
            `id_` 对应的{element_type_name_doc}。

        提示:
            若映射关系不包含 `id_` ，则本方法不改变映射关系，并返回 ``None`` 。
        \"\"\"
        element = self.__{element_type_name_member}_id_to_element.pop(
            str(id_), None)
        if element is not None:
            self.__{element_type_name_member}_element_to_id.pop(element)
            self.__{element_type_name_member}_element_instance_id.discard(
                id(element))
        return element

    def {element_type_name_member}_contains_id(self, id_: str) -> bool:
        \"\"\"判断{element_type_name_doc}的映射关系是否包含给定标识符。

        参数:
            id_: {element_type_name_doc}标识符。

        返回值:
            {element_type_name_doc}的映射关系是否包含 `id_` 。
        \"\"\"
        return str(id_) in self.__{element_type_name_member}_id_to_element

    def {element_type_name_member}_contains_instance(
            self, {element_type_name_member}: element_type) \
            -> bool:
        \"\"\"判断{element_type_name_doc}的映射关系是否包含给定{element_type_name}实
        例。

        参数:
            {element_type_name_member}: {element_type_name}实例。

        返回值:
            {element_type_name_doc}的映射关系是否包含
            `{element_type_name_member}` 。
        \"\"\"
        return \
            id({element_type_name_member}) in \
            self.__{element_type_name_member}_element_instance_id

    def {element_type_name_member}_get_iterator(self) \
            -> Iterator[element_type]:
        \"\"\"获取针对{element_type_name_doc}的迭代器。

        返回值:
            迭代器。每次迭代返回一个{element_type_name_doc}。
        \"\"\"
        return iter(self.__{element_type_name_member}_id_to_element.values())
        """,
        {
            **globals(), "element_type": element_type,
            "element_type_name_id_prefix": element_type_name_id_prefix
        },
        locals_)
    return locals_[element_type_name_class + "StatelessElementMap"]
