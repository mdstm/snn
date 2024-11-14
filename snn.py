'''SNN Dialect'''

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, i1
from xdsl.ir import (
  Block,
  BlockOps,
  Dialect,
  Operation,
  Region,
  SSAValue,
)
from xdsl import irdl
from xdsl.irdl import IRDLOperation, irdl_op_definition
from xdsl.traits import IsolatedFromAbove, NoTerminator

from xdslm import (
  DataAttr,
  Parser,
  Printer,
  attr_dict_to_py,
  attr_to_py,
  py_to_attr_dict,
)


# ------------------------------------------------------------------------------
# 模板类列表：
#   SNNOp_Symbol
#   SNNOp_Result
#   SNNOp_Operands
#
# 从 Op 中获取 mlir 属性用 op.attributes['<name>']
# 获取 python 格式所有属性用 op.get_py_attrs()
# 获取 python 格式单个属性用 op.get_<name>()
# 定义基类时可在基类中实现属性的 get 方法，在子类中实现具体处理方法
#
# Op 的 make 方法用于方便地用现有 python 格式属性构造 Op ，
# __init__ 只能接收封装好的 mlir 属性
#
# Op 返回值仅用于作为操作数指代对应的操作，其类型和具体值无意义
#
# 打印格式为：
# %0 = <name> @<sym_name> <attributes>
# %0 = <name> @<sym_name>(<operands>) <attributes>
# ------------------------------------------------------------------------------


class SNNOp_Symbol(IRDLOperation):
  '''所有 snn op 的基类，有一个符号名称'''

  sym_name: StringAttr = irdl.prop_def(StringAttr)

  def __init__(self,
    sym: str | StringAttr,
    attrs: Mapping[str, Any],
    **kwargs,
  ):
    super().__init__(
      properties={'sym_name': StringAttr(sym) if isinstance(sym, str) else sym},
      attributes=py_to_attr_dict(attrs),
      **kwargs,
    )

  @property
  def sym(self) -> str:
    '''符号名称'''
    return self.sym_name.data

  def get_py_attrs(self) -> dict[str, Any]:
    '''获取 python 格式所有属性'''
    return attr_dict_to_py(self.attributes)

  def __str__(self) -> str:
    from io import StringIO

    with StringIO() as f:
      Printer(f).print_op(self)
      return f.getvalue()


class SNNOp_Result(SNNOp_Symbol):
  '''没有操作数有一个返回值的 snn op'''

  res: irdl.OpResult = irdl.result_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    **attrs,
  ):
    super().__init__(sym, attrs, result_types=(i1,))

  @classmethod
  def parse(cls, p: Parser):
    # parse symbol_name
    sym = p.parse_symbol_name()

    # parse attributes
    attrs = p.parse_optional_attr_dict()

    return cls(sym, **attrs)

  def print(self, p: Printer):
    # print symbol_name
    p.print_string(f' @{self.sym}')

    # print attributes
    p.print_op_attributes(self.attributes)


class SNNOp_Operands(SNNOp_Symbol):
  '''有操作数和一个返回值的 snn op, 操作数及构造方法另行定义'''

  res: irdl.OpResult = irdl.result_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    *operands: Operation | SSAValue,
    **attrs,
  ):
    super().__init__(sym, attrs, operands=operands, result_types=(i1,))

  @classmethod
  def parse(cls, p: Parser):
    # parse symbol_name
    sym = p.parse_symbol_name()

    # parse operands
    operands = p.parse_comma_separated_list(p.Delimiter.PAREN, p.parse_operand)

    # parse attributes
    attrs = p.parse_optional_attr_dict()

    return cls(sym, *operands, **attrs)

  def print(self, p: Printer):
    # print symbol_name
    p.print_string(f' @{self.sym}')

    # print operands
    p.print_operands(self.operands)

    # print attributes
    p.print_op_attributes(self.attributes)


@irdl_op_definition
class Network(SNNOp_Symbol):
  '''SNN 模型'''

  name = 'snn.network'

  body: Region = irdl.region_def('single_block')

  traits = frozenset([IsolatedFromAbove(), NoTerminator()])

  def __init__(self,
    sym: str | StringAttr,
    body: Iterable[Operation] | Region | None = None,
    **attrs,
  ):
    # make body
    if not isinstance(body, Region):
      body = Region(Block()) if body is None else Region(Block(body))

    super().__init__(sym, attrs, regions=(body,))

  @property
  def ops(self) -> BlockOps:
    '''用于遍历所有内部 op'''
    return self.body.blocks[0].ops

  def add_op(self, op: Operation) -> Operation:
    '''添加 op'''
    self.body.blocks[0].add_op(op)
    return op

  @classmethod
  def parse(cls, p: Parser):
    # parse symbol_name
    sym = p.parse_symbol_name()

    # parse attributes
    attrs = p.parse_optional_attr_dict_with_keyword()
    attrs = {} if attrs is None else attrs.data

    # parse body
    body = p.parse_region()
    if not body.blocks:
      body.add_block(Block())

    return cls(sym, body, **attrs)

  def print(self, p: Printer):
    # print symbol_name
    p.print_string(f' @{self.sym}')

    # print attributes
    p.print_op_attributes(self.attributes, print_keyword=True)

    # print body
    p.print_string(' ')
    p.print_region(self.body, print_empty_block=False)


@irdl_op_definition
class EncoderModel(SNNOp_Result):
  '''脉冲编码器模型'''

  name = 'snn.encoder_model'

  def __init__(self,
    sym: str | StringAttr,
    parameter: Mapping[str, Any],
    **attrs,
  ):
    super().__init__(sym, parameter=parameter, **attrs)

  def get_param(self) -> dict[str, Any]:
    '''获取模型参数'''
    return attr_dict_to_py(self.attributes['parameter'].data)


@irdl_op_definition
class NeuronModel(SNNOp_Result):
  '''神经元模型'''

  name = 'snn.neuron_model'

  def __init__(self,
    sym: str | StringAttr,
    parameter: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    **attrs,
  ):
    super().__init__(
      sym,
      parameter=parameter,
      initial_state=initial_state,
      **attrs,
    )

  def get_param(self) -> dict[str, Any]:
    '''获取模型参数'''
    return attr_dict_to_py(self.attributes['parameter'].data)

  def get_state(self) -> dict[str, Any]:
    '''获取模型初始状态'''
    return attr_dict_to_py(self.attributes['initial_state'].data)


@irdl_op_definition
class TopologyModel(SNNOp_Result):
  '''连接组拓扑结构模型'''

  name = 'snn.topology_model'

  def __init__(self,
    sym: str | StringAttr,
    presynaptic_shape: Sequence[int] | ArrayAttr,
    postsynaptic_shape: Sequence[int] | ArrayAttr,
    parameter: Mapping[str, Any],
    **attrs,
  ):
    super().__init__(
      sym,
      presynaptic_shape=presynaptic_shape,
      postsynaptic_shape=postsynaptic_shape,
      parameter=parameter,
      **attrs,
    )

  def get_pre_shape(self) -> list[int]:
    '''获取前节点形状'''
    return [x.data for x in self.attributes['presynaptic_shape']]

  def get_post_shape(self) -> list[int]:
    '''获取后节点形状'''
    return [x.data for x in self.attributes['postsynaptic_shape']]

  def get_param(self) -> dict[str, Any]:
    '''获取模型参数'''
    return attr_dict_to_py(self.attributes['parameter'].data)


@irdl_op_definition
class SynapseModel(SNNOp_Result):
  '''突触模型'''

  name = 'snn.synapse_model'

  def __init__(self,
    sym: str | StringAttr,
    parameter: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    **attrs,
  ):
    super().__init__(
      sym,
      parameter=parameter,
      initial_state=initial_state,
      **attrs,
    )

  def get_param(self) -> dict[str, Any]:
    '''获取模型参数'''
    return attr_dict_to_py(self.attributes['parameter'].data)

  def get_state(self) -> dict[str, Any]:
    '''获取模型初始状态'''
    return attr_dict_to_py(self.attributes['initial_state'].data)


@irdl_op_definition
class DecoderModel(SNNOp_Result):
  '''解码器模型'''

  name = 'snn.decoder_model'

  def __init__(self,
    sym: str | StringAttr,
    parameter: Mapping[str, Any],
    **attrs,
  ):
    super().__init__(sym, parameter=parameter, **attrs)

  def get_param(self) -> dict[str, Any]:
    '''获取模型参数'''
    return attr_dict_to_py(self.attributes['parameter'].data)


@irdl_op_definition
class Encoder(SNNOp_Operands):
  '''脉冲编码器'''

  name = 'snn.encoder'

  model: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    encoder_model: EncoderModel | SSAValue,
    shape: Sequence[int] | ArrayAttr,
    **attrs,
  ):
    super().__init__(sym, encoder_model, shape=shape, **attrs)

  @property
  def model_op(self) -> EncoderModel:
    '''编码器模型 op'''
    return self.model.owner

  def get_shape(self) -> list[int]:
    '''获取形状'''
    return [x.data for x in self.attributes['shape']]


@irdl_op_definition
class NeuronGroup(SNNOp_Operands):
  '''神经元组'''

  name = 'snn.neuron_group'

  model: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    neuron_model: NeuronModel | SSAValue,
    shape: Sequence[int] | ArrayAttr,
    initial_state_value:
      Mapping[str, tuple[np.ndarray, bool]] | DictionaryAttr | None = None,
    **attrs,
  ):
    super().__init__(
      sym,
      neuron_model,
      shape=shape,
      initial_state_value=initial_state_value,
      **attrs,
    )

  @property
  def model_op(self) -> NeuronModel:
    '''神经元模型 op'''
    return self.model.owner

  def get_shape(self) -> list[int]:
    '''获取形状'''
    return [x.data for x in self.attributes['shape']]

  def get_state(self) -> dict[str, tuple[np.ndarray, bool]] | None:
    '''获取初始状态值'''
    return attr_to_py(self.attributes.get('initial_state_value'))


@irdl_op_definition
class ConnectionGroup(SNNOp_Operands):
  '''连接组'''

  name = 'snn.connection_group'

  pre: irdl.Operand = irdl.operand_def(i1)
  post: irdl.Operand = irdl.operand_def(i1)
  t_model: irdl.Operand = irdl.operand_def(i1)
  s_model: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    presynaptic_node: Encoder | NeuronGroup | SSAValue,
    postsynaptic_node: NeuronGroup | SSAValue,
    topology_model: TopologyModel | SSAValue,
    synapse_model: SynapseModel | SSAValue,
    initial_synapse_state_value:
      Mapping[str, tuple[np.ndarray, bool]] | DictionaryAttr | None = None,
    delay: float | DataAttr | None = None,
    **attrs,
  ):
    super().__init__(
      sym,
      presynaptic_node,
      postsynaptic_node,
      topology_model,
      synapse_model,
      initial_synapse_state_value=initial_synapse_state_value,
      delay=delay,
      **attrs,
    )

  @property
  def pre_op(self) -> Encoder | NeuronGroup:
    '''前节点 op'''
    return self.pre.owner

  @property
  def post_op(self) -> NeuronGroup:
    '''后节点 op'''
    return self.post.owner

  @property
  def t_model_op(self) -> TopologyModel:
    '''拓扑模型 op'''
    return self.t_model.owner

  @property
  def s_model_op(self) -> SynapseModel:
    '''突触模型 op'''
    return self.s_model.owner

  def get_state(self) -> dict[str, tuple[np.ndarray, bool]] | None:
    '''获取初始突触状态值'''
    return attr_to_py(self.attributes.get('initial_synapse_state_value'))

  def get_delay(self) -> float | None:
    '''获取延时'''
    return attr_to_py(self.attributes.get('delay'))


@irdl_op_definition
class SpikeDecoder(SNNOp_Operands):
  '''脉冲解码器'''

  name = 'snn.spike_decoder'

  model: irdl.Operand = irdl.operand_def(i1)
  target: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    decoder_model: DecoderModel | SSAValue,
    target_element: NeuronGroup | SSAValue,
    sampling_period: float | DataAttr,
    position: np.ndarray | DataAttr,
    **attrs,
  ):
    super().__init__(
      sym,
      decoder_model,
      target_element,
      sampling_period=sampling_period,
      position=position,
      **attrs,
    )

  @property
  def model_op(self) -> DecoderModel:
    '''解码器模型 op'''
    return self.model.owner

  @property
  def target_op(self) -> NeuronGroup:
    '''目标 op'''
    return self.target.owner

  def get_period(self) -> float:
    '''获取采样周期'''
    return self.attributes['sampling_period'].data

  def get_position(self) -> np.ndarray:
    '''获取观测位置'''
    return self.attributes['position'].data


@irdl_op_definition
class StateDecoderNeuron(SNNOp_Operands):
  '''神经元状态解码器'''

  name = 'snn.state_decoder_neuron'

  model: irdl.Operand = irdl.operand_def(i1)
  target: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    decoder_model: DecoderModel | SSAValue,
    target_element: NeuronGroup | SSAValue,
    sampling_period: float | DataAttr,
    position: np.ndarray | DataAttr,
    state_name: str | StringAttr,
    **attrs,
  ):
    super().__init__(
      sym,
      decoder_model,
      target_element,
      sampling_period=sampling_period,
      position=position,
      state_name=state_name,
      **attrs,
    )

  @property
  def model_op(self) -> DecoderModel:
    '''解码器模型 op'''
    return self.model.owner

  @property
  def target_op(self) -> NeuronGroup:
    '''目标 op'''
    return self.target.owner

  def get_period(self) -> float:
    '''获取采样周期'''
    return self.attributes['sampling_period'].data

  def get_position(self) -> np.ndarray:
    '''获取观测位置'''
    return self.attributes['position'].data

  def get_state_name(self) -> str:
    '''获取目标状态名称'''
    return self.attributes['state_name'].data


@irdl_op_definition
class StateDecoderSynapse(SNNOp_Operands):
  '''突触状态解码器'''

  name = 'snn.state_decoder_synapse'

  model: irdl.Operand = irdl.operand_def(i1)
  target: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    decoder_model: DecoderModel | SSAValue,
    target_element: ConnectionGroup | SSAValue,
    sampling_period: float | DataAttr,
    position: np.ndarray | DataAttr,
    state_name: str | StringAttr,
    **attrs,
  ):
    super().__init__(
      sym,
      decoder_model,
      target_element,
      sampling_period=sampling_period,
      position=position,
      state_name=state_name,
      **attrs,
    )

  @property
  def model_op(self) -> DecoderModel:
    '''解码器模型 op'''
    return self.model.owner

  @property
  def target_op(self) -> ConnectionGroup:
    '''目标 op'''
    return self.target.owner

  def get_period(self) -> float:
    '''获取采样周期'''
    return self.attributes['sampling_period'].data

  def get_position(self) -> np.ndarray:
    '''获取观测位置'''
    return self.attributes['position'].data

  def get_state_name(self) -> str:
    '''获取目标状态名称'''
    return self.attributes['state_name'].data


@irdl_op_definition
class SpikeMonitor(SNNOp_Operands):
  '''脉冲监视器'''

  name = 'snn.spike_monitor'

  target: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    target_element: NeuronGroup | SSAValue,
    sampling_period: float | DataAttr,
    position: np.ndarray | DataAttr,
    **attrs,
  ):
    super().__init__(
      sym,
      target_element,
      sampling_period=sampling_period,
      position=position,
      **attrs,
    )

  @property
  def target_op(self) -> NeuronGroup:
    '''目标 op'''
    return self.target.owner

  def get_period(self) -> float:
    '''获取采样周期'''
    return self.attributes['sampling_period'].data

  def get_position(self) -> np.ndarray:
    '''获取观测位置'''
    return self.attributes['position'].data


@irdl_op_definition
class StateMonitorNeuron(SNNOp_Operands):
  '''神经元状态监视器'''

  name = 'snn.state_monitor_neuron'

  target: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    target_element: NeuronGroup | SSAValue,
    sampling_period: float | DataAttr,
    position: np.ndarray | DataAttr,
    state_name: str | StringAttr,
    **attrs,
  ):
    super().__init__(
      sym,
      target_element,
      sampling_period=sampling_period,
      position=position,
      state_name=state_name,
      **attrs,
    )

  @property
  def target_op(self) -> NeuronGroup:
    '''目标 op'''
    return self.target.owner

  def get_period(self) -> float:
    '''获取采样周期'''
    return self.attributes['sampling_period'].data

  def get_position(self) -> np.ndarray:
    '''获取观测位置'''
    return self.attributes['position'].data

  def get_state_name(self) -> str:
    '''获取目标状态名称'''
    return self.attributes['state_name'].data


@irdl_op_definition
class StateMonitorSynapse(SNNOp_Operands):
  '''突触状态监视器'''

  name = 'snn.state_monitor_synapse'

  target: irdl.Operand = irdl.operand_def(i1)

  def __init__(self,
    sym: str | StringAttr,
    target_element: ConnectionGroup | SSAValue,
    sampling_period: float | DataAttr,
    position: np.ndarray | DataAttr,
    state_name: str | StringAttr,
    **attrs,
  ):
    super().__init__(
      sym,
      target_element,
      sampling_period=sampling_period,
      position=position,
      state_name=state_name,
      **attrs,
    )

  @property
  def target_op(self) -> ConnectionGroup:
    '''目标 op'''
    return self.target.owner

  def get_period(self) -> float:
    '''获取采样周期'''
    return self.attributes['sampling_period'].data

  def get_position(self) -> np.ndarray:
    '''获取观测位置'''
    return self.attributes['position'].data

  def get_state_name(self) -> str:
    '''获取目标状态名称'''
    return self.attributes['state_name'].data


SNN = Dialect('snn', [
  Network,

  EncoderModel,
  NeuronModel,
  TopologyModel,
  SynapseModel,
  DecoderModel,

  Encoder,
  NeuronGroup,
  ConnectionGroup,
  SpikeDecoder,
  StateDecoderNeuron,
  StateDecoderSynapse,
  SpikeMonitor,
  StateMonitorNeuron,
  StateMonitorSynapse,
])


ElementModel = (
  EncoderModel | NeuronModel | TopologyModel | SynapseModel
  | DecoderModel
)
'''模型元素'''

ElementNonModel = (
  Encoder | NeuronGroup | ConnectionGroup | SpikeDecoder
  | StateDecoderNeuron | StateDecoderSynapse | SpikeMonitor
  | StateMonitorNeuron | StateMonitorSynapse
)
'''非模型元素'''

Element = ElementModel | ElementNonModel
'''SNN 中所有元素的公共基类'''


# 测试

def main():
  import numpy as np

  from xdsl.builder import Builder
  from xdsl.ir import MLContext

  @Builder.implicit_region
  def body():
    # Encoder
    _0 = EncoderModel(
      sym='Poisson',
      parameter={
        'spike_rate_per_unit': 1.0,
        'time_step': 1.0,
      }
    )
    _1 = Encoder(
      sym='enc',
      encoder_model=_0,
      shape=(10,),
    )

    # NeuronGroup
    _2 = NeuronModel(
      sym='LIF',
      parameter={
        'capacitance': 8.0,
        'resistance': 1.0,
        'time_step': 1.0,
        'voltage_rest': np.int16(0),
        'threshold': np.int16(32767),
        'voltage_reset_value': np.int16(0),
        'refractory_period': 1.0,
        'voltage_initial': np.int16(0),
      },
      initial_state={},
    )
    _3 = NeuronGroup(
      sym='neg',
      neuron_model=_2,
      shape=(5,),
      initial_state_value={
        'weight_sum': (np.zeros((5,), np.int16), False),
        'voltage': (np.zeros((5,), np.int16), False),
      },
    )

    # Connection
    _4 = TopologyModel(
      sym='FullyConnected',
      presynaptic_shape=(10,),
      postsynaptic_shape=(5,),
      parameter={},
    )
    _5 = SynapseModel(
      sym='Delta',
      parameter={'bit_width_weight': 16},
      initial_state={'weight': np.int16(1)},
    )
    _6 = ConnectionGroup(
      sym='con',
      presynaptic_node=_1,
      postsynaptic_node=_3,
      topology_model=_4,
      synapse_model=_5,
      initial_synapse_state_value={
        'weight': (np.random.randint(-32768, 32768, (10, 5), np.int16), True),
      },
      delay=0.0,
    )

    # Decoder
    _7 = DecoderModel(
      sym='SpikeCount',
      parameter={},
    )
    _8 = StateDecoderNeuron(
      sym='dec',
      decoder_model=_7,
      target_element=_3,
      sampling_period=1.0,
      position=np.ones((5,), bool),
      state_name='voltage',
    )

    # Monitor
    _9 = StateMonitorSynapse(
      sym='mon',
      target_element=_6,
      sampling_period=1.0,
      position=np.ones((10, 5), bool),
      state_name='weight',
    )

  net_op = Network('tiny', body)

  with open('snn.mlir', 'w') as f:
    p = Printer(f)
    p.print_op(net_op)
    p.print_string('\n')

  ctx = MLContext()
  ctx.load_dialect(SNN)

  with open('snn.mlir', 'r') as f:
    net_op_ = Parser(ctx, f.read()).parse_op()

  print(str(net_op) == str(net_op_))


if __name__ == '__main__':
  main()
