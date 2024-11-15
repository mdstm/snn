import spaic

import snn
from spaic_to_wuyuan_info import get_infos


def spaic2mlir(net: spaic.Network) -> snn.Network:
  '''把 spaic 表示的 SNN 转换为 mlir'''

  infos = get_infos(net)

  net_op = snn.Network(net.name)

  # 存储已构造的编码器、神经元组和连接，用于构造连接、解码器和监视器
  elements: dict[str] = {}

  for name, info in infos.items():
    match info['type']:
      case 'Encoder':
        model_op = net_op.add_op(snn.EncoderModel(info['model_type'],
          **info['model_param']))
        op = net_op.add_op(snn.Encoder(name, model_op, **info['param']))
        elements[name] = op
      case 'NeuronGroup':
        model_op = net_op.add_op(snn.NeuronModel(info['model_type'],
          **info['model_param']))
        op = net_op.add_op(snn.NeuronGroup(name, model_op, **info['param']))
        elements[name] = op
      case 'ConnectionGroup':
        t_model_op = net_op.add_op(snn.TopologyModel(info['t_model_type'],
          **info['t_model_param']))
        s_model_op = net_op.add_op(snn.SynapseModel(info['s_model_type'],
          **info['s_model_param']))
        op = net_op.add_op(snn.ConnectionGroup(name,
          presynaptic_node=elements[info['pre']],
          postsynaptic_node=elements[info['post']],
          topology_model=t_model_op, synapse_model=s_model_op, **info['param']))
        elements[name] = op
      case 'SpikeDecoder':
        model_op = net_op.add_op(snn.DecoderModel(info['model_type'],
          **info['model_param']))
        net_op.add_op(snn.SpikeDecoder(name, decoder_model=model_op,
          target_element=elements[info['target']], **info['param']))
      case 'StateDecoderNeuron':
        model_op = net_op.add_op(snn.DecoderModel(info['model_type'],
          **info['model_param']))
        net_op.add_op(snn.StateDecoderNeuron(name, decoder_model=model_op,
          target_element=elements[info['target']], **info['param']))
      case 'SpikeMonitor':
        net_op.add_op(snn.SpikeMonitor(name, elements[info['target']],
          **info['param']))
      case 'StateMonitorNeuron':
        net_op.add_op(snn.StateMonitorNeuron(name, elements[info['target']],
          **info['param']))
      case 'StateMonitorSynapse':
        net_op.add_op(snn.StateMonitorSynapse(name, elements[info['target']],
          **info['param']))

  return net_op


# 测试

def main():
  # import yappi

  from quantize import quantize_net
  from test import test
  from tiny import TinyModel, ActorNetSpiking
  from xdslm import Printer


  @test('构造 spaic ')
  def net():
    return TinyModel()

  @test('量化')
  def _():
    quantize_net(net)

  # yappi.set_clock_type('cpu')
  # yappi.start()

  @test('转换 spaic -> mlir ')
  def net_op():
    return spaic2mlir(net)

  # with open('tiny_yappi.txt', 'w', encoding='utf-8') as f:
    # yappi.get_func_stats().print_all(f)
    # yappi.get_thread_stats().print_all(f)

  @test('打印 mlir ')
  def _():
    with open('spaic2mlir.mlir', 'w') as f:
      p = Printer(f)
      p.print_op(net_op)
      p.print_string('\n')

  return net_op


if __name__ == '__main__':
  main()
