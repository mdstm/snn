import numpy as np

import wuyuan.snn_model as wym

from .extracter import vars


def get_mon_info(a: wym.Monitor) -> dict:
  target = a.get_target_element()

  param = {
    'dt': float(a.get_sampling_period()),
  }
  info = {
    'type': 'StateMonitor',
    'target': target.get_element_id(),
    'param': param,
  }

  position = a.get_position()
  if isinstance(a, wym.SpikeMonitor):
    param['var_name'] = 'O'
    param['index'] = (
      tuple(position.T) if position.dtype != bool else
      'full' if np.all(position) else
      tuple(np.asarray(np.where(position), dtype=np.int32))
    )
  elif isinstance(a, wym.StateMonitorNeuron):
    param['var_name'], reshape = vars[
      target.get_neuron_model().__class__.__name__][a.get_state_name()]
    shape = target.get_shape()
    if position.dtype != bool:
      position_ = np.zeros(shape, dtype=bool) # 假设形状相同
      position_[tuple(position.T)] = True
      position = position_
    position = reshape(position, shape)[0] # 去除批数维度
    param['index'] = (
      'full' if np.all(position) else
      tuple(np.asarray(np.where(position), dtype=np.int32))
    )
  else:
    t_model = target.get_topology_model()
    param['var_name'], reshape = vars[t_model.__class__.__name__][
      a.get_state_name()]
    pre_shape = t_model.get_presynaptic_shape()
    post_shape = t_model.get_postsynaptic_shape()
    shape = t_model.get_shared_synapse_state_shape()
    if position.dtype != bool:
      position_ = np.zeros(shape, dtype=bool) # 形状都一样
      position_[tuple(position.T)] = True
      position = position_
    position = reshape(position, pre_shape, post_shape)
    param['index'] = (
      'full' if np.all(position) else
      tuple(np.asarray(np.where(position), dtype=np.int32))
    )

  return info
