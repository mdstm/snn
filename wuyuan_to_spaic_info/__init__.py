import wuyuan.snn_model as wym

from .encoder import get_enc_info
from .neuron import get_neg_info
from .connection import get_con_info
from .decoder import get_dec_info
from .monitor import get_mon_info


def get_infos(net_wy: wym.Network) -> dict:
  '''将 wuyuan 转换为 spaic 信息'''

  infos = {}

  # 解析编码器
  for name, a in net_wy._EncoderStatefulElementMap__encoder_id_to_element.items():
    infos[name] = get_enc_info(a)

  # 解析神经元组
  for name, a in net_wy._NeuronGroupStatefulElementMap__neuron_group_id_to_element.items():
    infos[name] = get_neg_info(a)

  # 解析连接
  for name, a in net_wy._ConnectionGroupStatefulElementMap__connection_group_id_to_element.items():
    infos[name] = get_con_info(a)

  # 解析解码器
  for name, a in net_wy._SpikeDecoderStatefulElementMap__spike_decoder_id_to_element.items():
    infos[name] = get_dec_info(a)

  for name, a in net_wy._StateDecoderNeuronStatefulElementMap__state_decoder_neuron_id_to_element.items():
    infos[name] = get_dec_info(a)

  # 解析监视器
  for name, a in net_wy._SpikeMonitorStatefulElementMap__spike_monitor_id_to_element.items():
    infos[name] = get_mon_info(a)

  for name, a in net_wy._StateMonitorNeuronStatefulElementMap__state_monitor_neuron_id_to_element.items():
    infos[name] = get_mon_info(a)

  for name, a in net_wy._StateMonitorSynapseStatefulElementMap__state_monitor_synapse_id_to_element.items():
    infos[name] = get_mon_info(a)

  return infos
