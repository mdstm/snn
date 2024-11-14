'''扩展 xdsl'''

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from xdsl.dialects.builtin import (
  ArrayAttr,
  BytesAttr,
  DictionaryAttr,
  StringAttr,
)
from xdsl.ir import Attribute, Data
from xdsl.irdl import irdl_attr_definition
from xdsl import parser
from xdsl import printer


@irdl_attr_definition
class DataAttr(Data):
  '''自定义内置数据属性'''
  name = 'data'

  @classmethod
  def parse_parameter(cls, parser):
    pass

  def print_parameter(self, printer):
    pass


class Printer(printer.Printer):
  '''修改了一些属性的打印格式'''

  TYPES = {
    'bool': 'i1',
    'int8': 'i8',
    'int16': 'i16',
    'int32': 'i32',
    'int64': 'i64',
    'uint8': 'ui8',
    'uint16': 'ui16',
    'uint32': 'ui32',
    'uint64': 'ui64',
    'float16': 'f16',
    'float32': 'f32',
    'float64': 'f64',
  }

  def print_data(self, data):
    '''打印数据属性中的数据'''

    if isinstance(data, np.number):
      self.print_string(' : '.join((str(data), self.TYPES[data.dtype.name])))

    elif isinstance(data, np.ndarray):
      if data.ndim == 0:
        return self.print_data(data.ravel()[0])

      if (type_name := self.TYPES[data.dtype.name]) == 'i1':
        data = data.astype(np.int8) # 将布尔值用数字表示

      data_str = (
        '' if data.size == 0 else
        str(data_0) if np.all(data == (data_0 := data.ravel()[0])) else
        str(data.tolist()) # 对全相等数组优化
      )
      self.print_string(''.join((
        'dense<', data_str, '> : tensor<',
        'x'.join(map(str, data.shape)), 'x', type_name, '>',
      )))

    elif isinstance(data, (bool, np.bool_)):
      self.print_string('true' if data else 'false')

    else:
      self.print_string(str(data))

  def print_attribute(self, attr: Attribute) -> None:
    '''增加了自定义数据属性的打印方式'''

    # 自定义的数据属性
    if isinstance(attr, DataAttr):
      self.print_data(attr.data)

    else:
      super().print_attribute(attr)

  def print_attr_dict(self, attr_dict: dict[str, Attribute]) -> None:
    '''用多行来打印属性字典'''

    if not attr_dict:
      return self.print_string('{}')

    self._indent += 1
    newline = ',\n' + ' ' * (self._indent * printer.indentNumSpaces)
    attr_iter = iter(attr_dict.items())

    # 左括号与第一个属性
    attr_name, attr = next(attr_iter)
    self.print_string(''.join(('{', newline[1:], attr_name, ' = ')))
    self.print_attribute(attr)

    # 剩下的属性
    for attr_name, attr in attr_iter:
      self.print_string(attr_name.join((newline, ' = ')))
      self.print_attribute(attr)

    # 右括号
    self.print_string(newline[1:-printer.indentNumSpaces] + '}')
    self._indent -= 1


class Parser(parser.Parser):
  '''修改了数字、密集属性的解析方式'''

  TYPES = {
    'i1': bool,
    'i8': np.int8,
    'i16': np.int16,
    'i32': np.int32,
    'i64': np.int64,
    'ui8': np.uint8,
    'ui16': np.uint16,
    'ui32': np.uint32,
    'ui64': np.uint64,
    'f16': np.float16,
    'f32': np.float32,
    'f64': np.float64,
  }

  def find_string(self, content: str, sub: str, begin: int):
    '''找一个字符串，返回找到的位置'''
    i = content.find(sub, begin)
    if i < 0:
      self.raise_error(f'Excepted "{sub}"')
    return i

  def _parse_builtin_dense_attr(self, _name) -> DataAttr:
    lexer = self.lexer
    content = lexer.input.content

    # 获取数据字符串位置
    data_begin = self.find_string(content, '<', lexer.pos - 1) + 1
    data_end = self.find_string(content, '>', data_begin)

    # 获取类型字符串位置，类型包含形状和数据类型
    type_begin = self.find_string(content, ':', data_end + 1)
    type_begin = self.find_string(content, 'tensor', type_begin + 1)
    type_begin = self.find_string(content, '<', type_begin + 6) + 1
    type_end = self.find_string(content, '>', type_begin)

    # 调整解析器位置
    lexer.pos = type_end + 1
    self._parser_state.current_token = lexer.lex()

    # 构造数据类型和形状
    type_list = content[type_begin:type_end].split('x')
    data_type = self.TYPES[type_list[-1]]
    shape = list(map(int, type_list[:-1]))

    # 构造数据
    data = np.fromstring(
      content[data_begin:data_end]
        .replace(' ', '').replace('[', '').replace(']', ''),
      dtype=data_type, sep=',',
    )
    data = data.reshape(shape) if data.size != 1 else np.full(shape, data)

    return DataAttr(data)

  def parse_optional_builtin_int_or_float_attr(self) -> DataAttr | None:
    state = self._parser_state

    if (value := self.parse_optional_boolean()) is not None:
      return DataAttr(value)

    if (value := self.parse_optional_number()) is None:
      return None

    # 没有类型则直接返回
    if state.current_token.kind.value != ':':
      return DataAttr(value)

    # 解析类型
    state.current_token = self.lexer.lex()
    if ((dtype := self.TYPES.get(state.current_token.text)) is None):
      self.raise_error('type error')
    state.current_token = self.lexer.lex()

    return DataAttr(dtype(value))


def py_to_attr_dict(data_dict: Mapping[str, Any]) -> dict[str, Attribute]:
  '''把 python 数据字典转换为 mlir 属性字典'''
  return {k: py_to_attr(v) for k, v in data_dict.items() if v is not None}


def py_to_attr(data: Any) -> Attribute:
  '''把 python 数据转换为 mlir 属性'''

  if isinstance(data, Attribute):
    return data

  if isinstance(data, (int, float, np.number, np.ndarray)):
    return DataAttr(data)

  if isinstance(data, str):
    return StringAttr(data)

  if isinstance(data, bytes):
    return BytesAttr(data)

  if isinstance(data, Mapping):
    return DictionaryAttr(py_to_attr_dict(data))

  if isinstance(data, Iterable):
    return ArrayAttr(map(py_to_attr, data))

  raise TypeError(f'{type(data)} 无法转换为 mlir 属性')


def attr_dict_to_py(attr_dict: Mapping[str, Any]) -> dict[str, Any]:
  '''把 mlir 属性字典转换为 python 数据字典'''
  return {k: attr_to_py(v) for k, v in attr_dict.items()}


def attr_to_py(attr: Any) -> Any:
  '''把 mlir 属性转换为 python 数据'''

  if not isinstance(attr, Attribute):
    return attr

  if isinstance(attr, (DataAttr, StringAttr, BytesAttr)):
    return attr.data

  if isinstance(attr, DictionaryAttr):
    return attr_dict_to_py(attr.data)

  if isinstance(attr, ArrayAttr):
    return list(map(attr_to_py, attr))

  raise TypeError(f'{type(attr)} 无法转换为 python 数据')
