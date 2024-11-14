import spaic

from spaic_to_wuyuan_info import get_infos
import wuyuan.snn_model as wym


def spaic2wuyuan(net: spaic.Network) -> wym.Network:
  '''把 spaic 表示的 SNN 转换为 wuyuan'''

  infos = get_infos(net)

  net_wy = wym.Network()

  # 存储已构造的编码器、神经元组和连接，用于构造连接、解码器和监视器
  elements: dict[str] = {}

  for name, info in infos.items():
    match info['type']:
      case 'Encoder':
        model = net_wy.encoder_model_get('0', getattr(wym.encoder_models,
          info['model_type'])(**info['model_param']))
        a = net_wy.encoder_add(wym.Encoder(model, **info['param']), name)
        elements[name] = a
      case 'NeuronGroup':
        model = net_wy.neuron_model_get('0', getattr(wym.neuron_models,
          info['model_type'])(**info['model_param']))
        a = net_wy.neuron_group_add(wym.NeuronGroup(model, **info['param']),
          name)
        elements[name] = a
      case 'ConnectionGroup':
        t_model = net_wy.topology_model_get('0', getattr(wym.topology_models,
          info['t_model_type'])(**info['t_model_param']))
        s_model = net_wy.synapse_model_get('0', getattr(wym.synapse_models,
          info['s_model_type'])(**info['s_model_param']))
        a = net_wy.connection_group_add(wym.ConnectionGroup(
          presynaptic_node=elements[info['pre']],
          postsynaptic_node=elements[info['post']],
          topology_model=t_model, synapse_model=s_model, **info['param']), name)
        elements[name] = a
      case 'SpikeDecoder':
        model = net_wy.decoder_model_get('0', getattr(wym.decoder_models,
          info['model_type'])(**info['model_param']))
        net_wy.spike_decoder_add(wym.SpikeDecoder(decoder_model=model,
          target_element=elements[info['target']], **info['param']), name)
      case 'StateDecoderNeuron':
        model = net_wy.decoder_model_get('0', getattr(wym.decoder_models,
          info['model_type'])(**info['model_param']))
        net_wy.state_decoder_neuron_add(wym.StateDecoderNeuron(
          decoder_model=model, target_element=elements[info['target']],
          **info['param']), name)
      case 'SpikeMonitor':
        net_wy.spike_monitor_add(wym.SpikeMonitor(elements[info['target']],
          **info['param']), name)
      case 'StateMonitorNeuron':
        net_wy.state_monitor_neuron_add(wym.StateMonitorNeuron(
          elements[info['target']], **info['param']), name)
      case 'StateMonitorSynapse':
        net_wy.state_monitor_synapse_add(wym.StateMonitorSynapse(
          elements[info['target']], **info['param']), name)

  return net_wy


# 测试

def main():
  from quantize import quantize_net
  from test import test
  from tiny import TinyModel, ActorNetSpiking

  @test('构造 spaic ')
  def net():
    return TinyModel()

  @test('量化')
  def _():
    quantize_net(net)

  @test('转换 spaic -> wuyuan ')
  def net_wy():
    return spaic2wuyuan(net)


if __name__ == '__main__':
  main()
