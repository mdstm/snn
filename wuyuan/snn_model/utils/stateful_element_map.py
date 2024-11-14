from collections.abc import Iterator

from ..element import Element
from .id_uniquer import IDUniquer


def create_stateful_element_map_class(
        element_type: type, element_type_name_id_prefix: str = None,
        element_type_name_class: str = None,
        element_type_name_member: str = None,
        element_type_name_doc: str = None) \
        -> type:
    """创建从标识符到有状态元素的映射关系类。

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
        创建的从标识符到有状态元素的映射关系类。

    提示:
        1. 本函数创建一个类，该类实现从标识符到 `element_type` 实例的映射关系。

        2. `element_type` 必须是Element类的派生类。
    """
    if not issubclass(element_type, Element):
        raise TypeError("element_type must be a type.")
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
class {element_type_name_class}StatefulElementMap:
    \"\"\"从标识符到{element_type_name_doc}的映射关系。\"\"\"
    def __init__(self):
        self.__{element_type_name_member}_id_to_element: \
            dict[str, element_type] = \
            dict()
        self.__{element_type_name_member}_element_instance_id: set[int] = set()
        self.__{element_type_name_member}_id_uniquer: IDUniquer = IDUniquer(
            self.{element_type_name_member}_contains_id,
            element_type_name_id_prefix)

    def {element_type_name_member}_add(
            self, {element_type_name_member}: element_type, id_: str = None) \
            -> element_type:
        \"\"\"添加从给定标识符到的给定{element_type_name_doc}的映射项。

        参数:
            {element_type_name_member}: {element_type_name_doc}。
            id_: {element_type_name_doc}标识符。

        返回值:
            已被添加到映射关系的{element_type_name_doc}。

        提示:
            1. 若 `id_` 不为 ``None`` 且不在映射关系中，则新映射项中的标识符为 `id_` ；
            否则，本方法生成一个不存在于映射关系中的标识符作为新映射项的标识符。

            2. 若 `{element_type_name_member}` 已被添加到映射关系，则本方法抛出
            ValueError。 

            3. 对于本方法返回的{element_type_name_doc}，本方法更新其标识符，使其与映射关
            系在本方法返回时的状态一致。
        \"\"\"
        if not isinstance({element_type_name_member}, element_type):
            raise TypeError(
                \"{element_type_name_member} must be a {element_type_name} \"
                \"instance.\")
        element_instance_id = id({element_type_name_member})
        if element_instance_id in \
                self.__{element_type_name_member}_element_instance_id:
            raise ValueError(
                \"{element_type_name_member} must be a {element_type_name} \"
                \"instance that is not in the {element_type_name_doc} \"
                \"stateful element map.\")
        if id_ is None:
            id__ = self.__{element_type_name_member}_id_uniquer.get()
        else:
            id__ = str(id_)
            if id__ in self.__{element_type_name_member}_id_to_element:
                id__ = self.__{element_type_name_member}_id_uniquer.get(id__)
        self.__{element_type_name_member}_id_to_element[id__] = \
            {element_type_name_member}
        try:
            self.__{element_type_name_member}_element_instance_id.add(
                element_instance_id)
        except:
            self.__{element_type_name_member}_id_to_element.pop(id__)
            raise
        {element_type_name_member}.set_element_id(id__)
        return {element_type_name_member}

    def {element_type_name_member}_remove(self, id_: str) -> element_type:
        \"\"\"移除给定标识符和对应的{element_type_name_doc}。

        参数:
            id_: {element_type_name_doc}标识符。

        返回值:
            `id_` 对应的{element_type_name_doc}。

        提示:
            若映射关系不包含 `id` ，则本方法返回 ``None`` 。
        \"\"\"
        element = self.__{element_type_name_member}_id_to_element.pop(
            str(id_), None)
        if element is not None:
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

    def {element_type_name_member}_get(self, id_: str) -> element_type:
        \"\"\"获取给定标识符对应的{element_type_name_doc}。

        参数:
            id_: {element_type_name_doc}标识符。

        返回值:
            `id_` 对应的{element_type_name_doc}。

        提示:
            若映射关系不包含 `id_` ，则本方法抛出KeyError。
        \"\"\"
        return self.__{element_type_name_member}_id_to_element[str(id_)]

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
    return locals_[element_type_name_class + "StatefulElementMap"]
