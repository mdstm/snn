import numpy as np

import wuyuan.snn_model as wym

from .extracter import Extracter, exts
from .util import update


def get_neg_info(a: wym.NeuronGroup) -> dict:
  info = {
    'type': 'NeuronGroup',
    'param': {
      'shape': a.get_shape(),
    },
    'backend': {},
  }
  model = a.get_neuron_model()
  return update(info, exts[model.__class__.__name__](a, model))


class LIF(Extracter):
  def reshape(arr: np.ndarray, shape: list[int]) -> np.ndarray:
    return arr[None, ...]

  var_dict = {
    'weight_sum': ('Isyn', reshape),
    'voltage': ('V', reshape),
  }

  def get_info(a: wym.NeuronGroup, model: wym.NeuronModel) -> dict:
    p = model._ElementModelParameter__parameter
    param = {}
    backend = [
      ('Vth', p['threshold']),
    ]
    info = {
      'param': param,
      'backend': backend,
    }

    if p['time_step'] > 1e-5:
      param['model'] = 'lif'
      param['tau_m'] = p['capacitance'] * p['resistance'] # spaic neg dt 没用
      backend.append(('Vreset', p['voltage_reset_value'])) # 只能处理一种情况
    else:
      param['model'] = 'if'
      backend.append(('ConstantDecay', p['voltage_rest']))

    s0 = a._NeuronGroup__initial_neuron_state_value
    s1 = model._ElementModelState__state
    shape = a.get_shape() # 物源神经元组所有状态形状都一样，出问题也有个高的顶着
    backend.extend((var, reshape(
      value[0] if (value := s0.get(state_name)) is not None else
      np.full(shape, s1[state_name]), shape,
    )) for state_name, (var, reshape) in LIF.var_dict.items())

    return info


class CLIF(Extracter):
  var_dict = {
    'weight_sum': ('Isyn', LIF.reshape),
    'voltage_m': ('M', LIF.reshape),
    'voltage_s': ('S', LIF.reshape),
    'voltage_e': ('E', LIF.reshape),
  }

  def get_info(a: wym.NeuronGroup, model: wym.NeuronModel) -> dict:
    p = model._ElementModelParameter__parameter
    backend = [
      ('Vth', p['threshold']),
    ]
    info = {
      'param': {
        'model': 'clif',
        'tau_p': p['tau_m'],
        'tau_q': p['tau_s'],
        'tau_m': p['tau_e'],
      },
      'backend': backend,
    }

    s0 = a._NeuronGroup__initial_neuron_state_value
    s1 = model._ElementModelState__state
    shape = a.get_shape()
    backend.extend((var, reshape(
      value[0] if (value := s0.get(state_name)) is not None else
      np.full(shape, s1[state_name]), shape,
    )) for state_name, (var, reshape) in CLIF.var_dict.items())

    return info
