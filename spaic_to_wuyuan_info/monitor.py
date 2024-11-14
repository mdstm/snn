import numpy as np
import spaic

from .extracter import vars
from .util import get_value


def get_mon_info(a: spaic.StateMonitor, infos: dict) -> dict:
  target = a.target
  target_name = target.name
  target_info = infos[target_name]
  target_type = target_info['type']
  if target_type not in {'NeuronGroup', 'ConnectionGroup'}:
    raise TypeError('Monitor 目标只能是 NeuronGroup 或 ConnectionGroup')

  # 默认信息
  param = {
    'sampling_period': float(a.dt),
  }
  info = {
    'target': target_name,
    'param': param,
  }

  # 设置状态名称和观测位置
  var_name = a.var_name # spaic Monitor 会把简单名称变成长名称
  i = var_name.find('{') + 1
  var = var_name[i:var_name.find('}', i)]
  index = a.index
  if var == 'O':
    info['type'] = 'SpikeMonitor'
    if index == 'full':
      param['position'] = np.ones(target_info['param']['shape'], dtype=bool)
    else:
      position = np.zeros(target_info['param']['shape'], dtype=bool)
      # spaic 允许多一个批数维度，忽略
      position[index if len(index) == position.ndim else index[1:]] = True
      param['position'] = position
  elif target_type == 'NeuronGroup':
    info['type'] = 'StateMonitorNeuron'
    state_name, reshape = vars[target.model.__class__.__name__][var]
    param['state_name'] = state_name
    if index == 'full':
      param['position'] = np.ones_like(
        target_info['param']['initial_state_value'][state_name][0], dtype=bool,
      )
    else:
      # 先从后端获取原始形状，设置位置后再转换为 wuyuan 形状
      position = np.zeros_like(get_value(target, var_name)[0], dtype=bool)
      position[index if len(index) == position.ndim else index[1:]] = True
      param['position'] = reshape(
        position[None, ...], target_info['param']['shape'],
      )
  else:
    info['type'] = 'StateMonitorSynapse'
    state_name, reshape = vars[target.__class__.__name__][var]
    param['state_name'] = state_name
    if index == 'full':
      param['position'] = np.ones_like(
        target_info['param']['initial_synapse_state_value'][state_name][0],
        dtype=bool,
      )
    else:
      position = np.zeros_like(get_value(target, var_name), dtype=bool)
      position[index] = True
      target_t_model_param = target_info['t_model_param']
      param['position'] = reshape(
        position,
        target_t_model_param['presynaptic_shape'],
        target_t_model_param['postsynaptic_shape'],
      )

  return info
