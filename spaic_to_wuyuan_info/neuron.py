import numpy as np
import spaic

from .extracter import Extracter, exts
from .util import get_neg_value as get, update


def get_neg_info(a: spaic.NeuronGroup) -> dict:
  info = {
    'type': 'NeuronGroup',
    'param': {
      'shape': list(a.shape),
      'initial_state_value': {},
    },
    'model_type': 'unknown',
    'model_param': {
      'parameter': {},
      'initial_state': {},
    },
  }
  return update(info, exts[a.model.__class__.__name__](a))


class LIFModel(Extracter):
  def reshape(arr: np.ndarray, shape: list[int]) -> np.ndarray:
    '''神经元组变量默认的形状变化函数。spaic 后端会加一个批数维度。'''
    return arr[0]

  var_dict = {
    'Isyn': ('weight_sum', reshape),
    'V': ('voltage', reshape),
  }
  '''spaic 到 wuyuan 的变量名称及形状对应关系'''

  def get_info(a: spaic.NeuronGroup) -> dict:
    # spaic 的神经元组虽然有自己的 dt，但将 tau 值放入后端会使用后端的 dt
    dt = float(a._backend.dt)
    shape = a.shape
    return {
      'model_type': 'LIF',
      'model_param': {
        'parameter': {
          'capacitance': float(a.model._tau_variables['tauM']), # tauM = RC
          'resistance': 1.0,
          'time_step': dt,
          'voltage_rest': np.int16(0),
          'threshold': np.int16(get(a, 'Vth')), # 电位变量需要从后端获取量化值
          'voltage_reset_value': np.int16(get(a, 'Vreset')),
          'refractory_period': dt, # spaic 没有不应期，设为 dt
          'voltage_initial': np.int16(0),
        },
      },
      'param': {
        'initial_state_value': {
          state_name: (reshape(get(a, var), shape).astype(np.int16), False)
            for var, (state_name, reshape) in LIFModel.var_dict.items()
        },
      },
    }


class IFModel(Extracter):
  var_dict = LIFModel.var_dict

  def get_info(a: spaic.NeuronGroup) -> dict:
    shape = a.shape
    return {
      'model_type': 'LIF',
      'model_param': {
        'parameter': {
          'capacitance': 1.0,
          'resistance': 1.0,
          'time_step': 0.0,
          'voltage_rest': np.int16(get(a, 'ConstantDecay')), # 借用空闲位置
          'threshold': np.int16(get(a, 'Vth')),
          'voltage_reset_value': np.int16(0),
          'refractory_period': float(a._backend.dt),
          'voltage_initial': np.int16(0),
        },
      },
      'param': {
        'initial_state_value': {
          state_name: (reshape(get(a, var), shape).astype(np.int16), False)
            for var, (state_name, reshape) in LIFModel.var_dict.items()
        },
      },
    }


class CLIFModel(Extracter):
  var_dict = {
    'Isyn': ('weight_sum', LIFModel.reshape),
    'M': ('voltage_m', LIFModel.reshape),
    'S': ('voltage_s', LIFModel.reshape),
    'E': ('voltage_e', LIFModel.reshape),
  }

  def get_info(a: spaic.NeuronGroup) -> dict:
    tau_variables = a.model._tau_variables
    shape = a.shape
    return {
      'model_type': 'CLIF',
      'model_param': {
        'parameter': {
          'tau_m': float(tau_variables['tauP']),
          'tau_s': float(tau_variables['tauQ']),
          'tau_e': float(tau_variables['tauM']),
          'time_step': float(a._backend.dt),
          'threshold': np.int16(get(a, 'Vth')),
        },
      },
      'param': {
        'initial_state_value': {
          state_name: (reshape(get(a, var), shape).astype(np.int16), False)
            for var, (state_name, reshape) in CLIFModel.var_dict.items()
        },
      },
    }
