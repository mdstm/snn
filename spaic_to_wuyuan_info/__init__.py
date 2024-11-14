import spaic

from .encoder import get_enc_info
from .neuron import get_neg_info
from .connection import get_con_info
from .decoder import get_dec_info
from .monitor import get_mon_info


def get_infos(net: spaic.Network) -> dict:
  '''将 spaic 转换为 wuyuan 信息'''

  infos = {}

  # 解析编码器
  for name, a in net._groups.items():
    if isinstance(a, spaic.Encoder):
      infos[name] = get_enc_info(a)

  # 解析神经元组
  for name, a in net._groups.items():
    if isinstance(a, spaic.NeuronGroup):
      infos[name] = get_neg_info(a)

  # 解析连接
  for name, a in net._connections.items():
    infos[name] = get_con_info(a, infos)

  # 解析解码器
  for name, a in net._groups.items():
    if isinstance(a, spaic.Decoder):
      infos[name] = get_dec_info(a, infos)

  # 解析监视器
  for name, a in net._monitors.items():
    infos[name] = get_mon_info(a, infos)

  return infos
