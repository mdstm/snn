import numpy as np
import spaic

from .extracter import Extracter, exts
from .util import get_con_value as get, update


def get_con_info(a: spaic.Connection, infos: dict) -> dict:
  pre_name, post_name = a.pre.name, a.post.name
  pre_shape = infos[pre_name]['param']['shape']
  post_shape = infos[post_name]['param']['shape']
  info = {
    'type': 'ConnectionGroup',
    'pre': pre_name,
    'post': post_name,
    'param': {
      'initial_synapse_state_value': {},
      'delay': float(a.max_delay),
    },
    't_model_type': 'unknown',
    't_model_param': {
      'presynaptic_shape': pre_shape,
      'postsynaptic_shape': post_shape,
      'parameter': {},
    },
    's_model_type': 'Delta',
    's_model_param': {
      'parameter': {'bit_width_weight': 16},
      'initial_state': {'weight': np.int16(1)},
    },
  }
  return update(info, exts[a.__class__.__name__](a, pre_shape, post_shape))


class FullConnection(Extracter):
  def reshape(arr: np.ndarray,
      pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
    '''spaic: (post_num, pre_num) -> wuyuan: pre_shape + post_shape'''
    return arr.T.reshape(pre_shape + post_shape)

  var_dict = {
    'weight': ('weight', reshape),
  }

  def get_info(a: spaic.Connection,
      pre_shape: list[int], post_shape: list[int]) -> dict:
    return {
      't_model_type': 'FullyConnected',
      'param': {
        'initial_synapse_state_value': {
          state_name: (
            reshape(get(a, var), pre_shape, post_shape).astype(np.int16), True
          ) for var, (state_name, reshape) in FullConnection.var_dict.items()
        },
      },
    }


class one_to_one_mask(Extracter):
  def reshape(arr: np.ndarray,
      pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
    '''spaic 把权重放在对角线上'''
    return arr.diagonal()

  var_dict = {
    'weight': ('weight', reshape),
  }

  def get_info(a: spaic.Connection,
      pre_shape: list[int], post_shape: list[int]) -> dict:
    return {
      't_model_type': 'OneToOne',
      'param': {
        'initial_synapse_state_value': {
          state_name: (
            reshape(get(a, var), pre_shape, post_shape).astype(np.int16), True
          ) for var, (state_name, reshape) in one_to_one_mask.var_dict.items()
        },
      },
    }


class conv_connect(Extracter):
  def reshape(arr: np.ndarray,
      pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
    '''卷积都一样'''
    return arr

  var_dict = {
    'weight': ('weight', reshape),
  }

  def get_info(a: spaic.Connection,
      pre_shape: list[int], post_shape: list[int]) -> dict:
    return {
    't_model_type': 'Convolution2D',
    't_model_param': {
      'parameter': {
        'kernel_size': a.kernel_size,
        'padding': a.padding,
        'stride': a.stride,
        'dilation': (1, 1), # spaic 没有用到膨胀系数
      },
    },
    'param': {
      'initial_synapse_state_value': {
        state_name: (
          reshape(get(a, var), pre_shape, post_shape).astype(np.int16), True
        ) for var, (state_name, reshape) in conv_connect.var_dict.items()
      },
    },
  }
