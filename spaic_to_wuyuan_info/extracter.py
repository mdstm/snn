from typing import Callable


exts: dict[str, Callable] = {}
vars: dict[str, tuple[str, Callable]] = {}


class Meta(type):
  def __new__(cls, name, *args):
    a = type.__new__(cls, name, *args)
    if (get_info := getattr(a, 'get_info', None)) is not None:
      exts[name] = get_info
    if (var_dict := getattr(a, 'var_dict', None)) is not None:
      vars[name] = var_dict
    return a


class Extracter(metaclass=Meta):
  '''继承该类会自动将两种属性添加到字典里'''
  pass
