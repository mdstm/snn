import numpy as np
import spaic

from .extracter import Extracter, exts, vars
from .util import update


def get_dec_info(a: spaic.Decoder, infos: dict) -> dict:
  target = a.dec_target
  target_name = target.name
  target_info = infos[target_name]
  if target_info['type'] != 'NeuronGroup':
    raise TypeError('Decoder 目标只能是 NeuronGroup')

  # 默认信息
  param = {
    'sampling_period': float(a.dt),
  }
  info = {
    'target': target_name,
    'param': param,
    'model_type': 'unknown',
    'model_param': {
      'parameter': {},
    },
  }

  # 设置状态名称和观测位置
  var = a.coding_var_name
  if var == 'O':
    info['type'] = 'SpikeDecoder'
    param['position'] = np.ones(target_info['param']['shape'], dtype=bool)
  else:
    info['type'] = 'StateDecoderNeuron'
    state_name, _ = vars[target.model.__class__.__name__][var]
    param['state_name'] = state_name
    param['position'] = np.ones_like(
      target_info['param']['initial_state_value'][state_name][0], dtype=bool,
    )

  return update(info, exts[a.__class__.__name__](a))


class Spike_Counts(Extracter):
  def get_info(a: spaic.Decoder) -> dict:
    return {
      'model_type': 'SpikeCount',
    }
