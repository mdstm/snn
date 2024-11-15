import wuyuan.snn_model as wym
import snn


def wuyuan2mlir(name: str, net_wy: wym.Network) -> snn.Network:
  '''把 wuyuan 表示的 SNN 转换为 mlir'''

  net_op = snn.Network(name)

  # 存储已构造的编码器、神经元组和连接，用于构造连接、解码器和监视器
  elements: dict[str] = {}

  # 解析编码器
  for name, a in net_wy._EncoderStatefulElementMap__encoder_id_to_element.items():
    model = a.get_encoder_model()
    model_op = net_op.add_op(snn.EncoderModel(
      sym=model.__class__.__name__,
      parameter=model._ElementModelParameter__parameter,
    ))
    op = net_op.add_op(snn.Encoder(
      sym=name,
      encoder_model=model_op,
      shape=a.get_shape(),
    ))
    elements[name] = op

  # 解析神经元组
  for name, a in net_wy._NeuronGroupStatefulElementMap__neuron_group_id_to_element.items():
    model = a.get_neuron_model()
    model_op = net_op.add_op(snn.NeuronModel(
      sym=model.__class__.__name__,
      parameter=model._ElementModelParameter__parameter,
      initial_state=model._ElementModelState__state,
    ))
    op = net_op.add_op(snn.NeuronGroup(
      sym=name,
      neuron_model=model_op,
      shape=a.get_shape(),
      initial_state_value=a._NeuronGroup__initial_neuron_state_value,
    ))
    elements[name] = op

  # 解析连接
  for name, a in net_wy._ConnectionGroupStatefulElementMap__connection_group_id_to_element.items():
    t_model = a.get_topology_model()
    t_model_op = net_op.add_op(snn.TopologyModel(
      sym=t_model.__class__.__name__,
      presynaptic_shape=t_model.get_presynaptic_shape(),
      postsynaptic_shape=t_model.get_postsynaptic_shape(),
      parameter=t_model._ElementModelParameter__parameter,
    ))
    s_model = a.get_synapse_model()
    s_model_op = net_op.add_op(snn.SynapseModel(
      sym=s_model.__class__.__name__,
      parameter=s_model._ElementModelParameter__parameter,
      initial_state=s_model._ElementModelState__state,
    ))
    op = net_op.add_op(snn.ConnectionGroup(
      sym=name,
      presynaptic_node=elements[a.get_presynaptic_node().get_element_id()],
      postsynaptic_node=elements[a.get_postsynaptic_node().get_element_id()],
      topology_model=t_model_op,
      synapse_model=s_model_op,
      initial_synapse_state_value=a._ConnectionGroup__initial_synapse_state_value,
      delay=a.get_delay(),
    ))
    elements[name] = op

  # 解析解码器
  for name, a in net_wy._SpikeDecoderStatefulElementMap__spike_decoder_id_to_element.items():
    model = a.get_decoder_model()
    model_op = net_op.add_op(snn.DecoderModel(
      sym=model.__class__.__name__,
      parameter=model._ElementModelParameter__parameter,
    ))
    net_op.add_op(snn.SpikeDecoder(
      sym=name,
      decoder_model=model_op,
      target_element=elements[a.get_target_element().get_element_id()],
      sampling_period=a.get_sampling_period(),
      position=a.get_position(),
    ))

  for name, a in net_wy._StateDecoderNeuronStatefulElementMap__state_decoder_neuron_id_to_element.items():
    model = a.get_decoder_model()
    model_op = net_op.add_op(snn.DecoderModel(
      sym=model.__class__.__name__,
      parameter=model._ElementModelParameter__parameter,
    ))
    net_op.add_op(snn.StateDecoderNeuron(
      sym=name,
      decoder_model=model_op,
      target_element=elements[a.get_target_element().get_element_id()],
      sampling_period=a.get_sampling_period(),
      position=a.get_position(),
      state_name=a.get_state_name(),
    ))

  for name, a in net_wy._StateDecoderSynapseStatefulElementMap__state_decoder_synapse_id_to_element.items():
    model = a.get_decoder_model()
    model_op = net_op.add_op(snn.DecoderModel(
      sym=model.__class__.__name__,
      parameter=model._ElementModelParameter__parameter,
    ))
    net_op.add_op(snn.StateDecoderSynapse(
      sym=name,
      decoder_model=model_op,
      target_element=elements[a.get_target_element().get_element_id()],
      sampling_period=a.get_sampling_period(),
      position=a.get_position(),
      state_name=a.get_state_name(),
    ))

  # 解析监视器
  for name, a in net_wy._SpikeMonitorStatefulElementMap__spike_monitor_id_to_element.items():
    net_op.add_op(snn.SpikeMonitor(
      sym=name,
      target_element=elements[a.get_target_element().get_element_id()],
      sampling_period=a.get_sampling_period(),
      position=a.get_position(),
    ))

  for name, a in net_wy._StateMonitorNeuronStatefulElementMap__state_monitor_neuron_id_to_element.items():
    net_op.add_op(snn.StateMonitorNeuron(
      sym=name,
      target_element=elements[a.get_target_element().get_element_id()],
      sampling_period=a.get_sampling_period(),
      position=a.get_position(),
      state_name=a.get_state_name(),
    ))

  for name, a in net_wy._StateMonitorSynapseStatefulElementMap__state_monitor_synapse_id_to_element.items():
    net_op.add_op(snn.StateMonitorSynapse(
      sym=name,
      target_element=elements[a.get_target_element().get_element_id()],
      sampling_period=a.get_sampling_period(),
      position=a.get_position(),
      state_name=a.get_state_name(),
    ))

  return net_op


# 测试

def main():
  from xdsl.ir import MLContext

  from mlir2wuyuan import mlir2wuyuan
  from test import test
  from xdslm import Parser, Printer

  ctx = MLContext()
  ctx.load_dialect(snn.SNN)

  @test('读取 mlir ')
  def text():
    with open('spaic2mlir.mlir', 'r') as f:
      return f.read()

  @test('解析 mlir ')
  def net_op():
    return Parser(ctx, text).parse_op()

  @test('转换 mlir -> wuyuan ')
  def net_wy():
    return mlir2wuyuan(net_op)

  @test('转换 wuyuan -> mlir ')
  def net_op_():
    return wuyuan2mlir(net_op.sym, net_wy)

  @test('打印 mlir')
  def _():
    with open('wuyuan2mlir.mlir', 'w') as f:
      p = Printer(f)
      p.print_op(net_op_)
      p.print_string('\n')

  return net_op, net_wy, net_op_


if __name__ == '__main__':
  main()
