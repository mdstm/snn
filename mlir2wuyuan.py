import wuyuan.snn_model as wym
import snn


def mlir2wuyuan(net_op: snn.Network) -> wym.Network:
  '''把 mlir 表示的 SNN 转换为 wym'''

  net_wy = wym.Network()

  # 存储已构造的编码器、神经元组和连接，用于构造连接、解码器和监视器
  elements: dict[str] = {}

  # 解析编码器和神经元组
  for op in net_op.ops:
    if isinstance(op, snn.Encoder):
      model_op = op.model_op
      model = net_wy.encoder_model_get('0', getattr(wym.encoder_models,
        model_op.sym)(**model_op.get_py_attrs()))
      name = op.sym
      a = net_wy.encoder_add(wym.Encoder(model, **op.get_py_attrs()), name)
      elements[name] = a
    elif isinstance(op, snn.NeuronGroup):
      model_op = op.model_op
      model = net_wy.neuron_model_get('0', getattr(wym.neuron_models,
        model_op.sym)(**model_op.get_py_attrs()))
      name = op.sym
      a = net_wy.neuron_group_add(wym.NeuronGroup(model, **op.get_py_attrs()),
        name)
      elements[name] = a

  # 解析连接
  for op in net_op.ops:
    if isinstance(op, snn.ConnectionGroup):
      t_model_op = op.t_model_op
      t_model = net_wy.topology_model_get('0', getattr(wym.topology_models,
        t_model_op.sym)(**t_model_op.get_py_attrs()))
      s_model_op = op.s_model_op
      s_model = net_wy.synapse_model_get('0', getattr(wym.synapse_models,
        s_model_op.sym)(**s_model_op.get_py_attrs()))
      name = op.sym
      a = net_wy.connection_group_add(wym.ConnectionGroup(
        presynaptic_node=elements[op.pre_op.sym],
        postsynaptic_node=elements[op.post_op.sym],
        topology_model=t_model, synapse_model=s_model, **op.get_py_attrs(),
      ), name)
      elements[name] = a

  # 解析解码器和监视器
  for op in net_op.ops:
    if isinstance(op, snn.SpikeDecoder):
      model_op = op.model_op
      model = net_wy.decoder_model_get('0', getattr(wym.decoder_models,
        model_op.sym)(**model_op.get_py_attrs()))
      net_wy.spike_decoder_add(wym.SpikeDecoder(decoder_model=model,
        target_element=elements[op.target_op.sym], **op.get_py_attrs()), op.sym)
    if isinstance(op, snn.StateDecoderNeuron):
      model_op = op.model_op
      model = net_wy.decoder_model_get('0', getattr(wym.decoder_models,
        model_op.sym)(**model_op.get_py_attrs()))
      net_wy.state_decoder_neuron_add(wym.StateDecoderNeuron(
        decoder_model=model, target_element=elements[op.target_op.sym],
        **op.get_py_attrs()), op.sym)
    if isinstance(op, snn.StateDecoderSynapse):
      model_op = op.model_op
      model = net_wy.decoder_model_get('0', getattr(wym.decoder_models,
        model_op.sym)(**model_op.get_py_attrs()))
      net_wy.state_decoder_synapse_add(wym.StateDecoderSynapse(
        decoder_model=model, target_element=elements[op.target_op.sym],
        **op.get_py_attrs()), op.sym)
    elif isinstance(op, snn.SpikeMonitor):
      net_wy.spike_monitor_add(wym.SpikeMonitor(elements[op.target_op.sym],
        **op.get_py_attrs()), op.sym)
    elif isinstance(op, snn.StateMonitorNeuron):
      net_wy.state_monitor_neuron_add(wym.StateMonitorNeuron(
        target_element=elements[op.target_op.sym], **op.get_py_attrs()), op.sym)
    elif isinstance(op, snn.StateMonitorSynapse):
      net_wy.state_monitor_synapse_add(wym.StateMonitorSynapse(
        target_element=elements[op.target_op.sym], **op.get_py_attrs()), op.sym)

  return net_wy


# 测试

def main():
  from xdsl.ir import MLContext

  from test import test
  from xdslm import Parser

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

  return net_op, net_wy


if __name__ == '__main__':
  main()
