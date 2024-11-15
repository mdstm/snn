import math

import numpy as np

import wuyuan.snn_model as wym

from .extracter import Extracter, exts
from .util import update


def get_con_info(a: wym.ConnectionGroup) -> dict:
  info = {
    'type': 'Connection',
    'pre': a.get_presynaptic_node().get_element_id(),
    'post': a.get_postsynaptic_node().get_element_id(),
    'param': {
      'max_delay': a.get_delay(),
    },
  }
  t_model = a.get_topology_model()
  s_model = a.get_synapse_model()
  return update(info, exts[t_model.__class__.__name__](a, t_model, s_model))


class FullyConnected(Extracter):
  def reshape(arr: np.ndarray,
      pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
    return arr.reshape((math.prod(pre_shape), math.prod(post_shape))).T

  var_dict = {
    'weight': ('weight', reshape),
  }

  def get_info(a: wym.ConnectionGroup,
      t_model: wym.TopologyModel, s_model: wym.SynapseModel) -> dict:
    pre_shape = t_model.get_presynaptic_shape()
    post_shape = t_model.get_postsynaptic_shape()
    shape = t_model.get_shared_synapse_state_shape() # 形状都一样
    s0 = a._ConnectionGroup__initial_synapse_state_value
    s1 = s_model._ElementModelState__state
    param = {
      'link_type': 'full',
    }
    if len(pre_shape) > 1:
      param['syn_type'] = ['flatten', 'basic']
    return {
      'param': param,
      'backend': [(var, reshape(
        value[0] if (value := s0.get(state_name)) is not None else
        np.full(shape, s1[state_name]), pre_shape, post_shape,
      )) for state_name, (var, reshape) in FullyConnected.var_dict.items()]
    }


class OneToOne(Extracter):
  def reshape(arr: np.ndarray,
      pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
    return np.diag(arr)

  var_dict = {
    'weight': ('weight', reshape),
  }

  def get_info(a: wym.ConnectionGroup,
      t_model: wym.TopologyModel, s_model: wym.SynapseModel) -> dict:
    pre_shape = t_model.get_presynaptic_shape()
    post_shape = t_model.get_postsynaptic_shape()
    shape = t_model.get_shared_synapse_state_shape()
    s0 = a._ConnectionGroup__initial_synapse_state_value
    s1 = s_model._ElementModelState__state
    return {
      'param': {
        'link_type': 'one_to_one',
      },
      'backend': [(var, reshape(
        value[0] if (value := s0.get(state_name)) is not None else
        np.full(shape, s1[state_name]), pre_shape, post_shape,
      )) for state_name, (var, reshape) in OneToOne.var_dict.items()]
    }


class Convolution2D(Extracter):
  def reshape(arr: np.ndarray,
      pre_shape: list[int], post_shape: list[int]) -> np.ndarray:
    return arr

  var_dict = {
    'weight': ('weight', reshape),
  }

  def get_info(a: wym.ConnectionGroup,
      t_model: wym.TopologyModel, s_model: wym.SynapseModel) -> dict:
    p = t_model._ElementModelParameter__parameter
    pre_shape = t_model.get_presynaptic_shape()
    post_shape = t_model.get_postsynaptic_shape()
    shape = t_model.get_shared_synapse_state_shape()
    s0 = a._ConnectionGroup__initial_synapse_state_value
    s1 = s_model._ElementModelState__state
    padding = p['padding']
    return {
      'param': {
        'link_type': 'conv',
        'in_channels': pre_shape[0],
        'kernel_size': p['kernel_size'],
        'padding': (padding[0], padding[2]),
        'stride': p['stride'],
      },
      'backend': [(var, reshape(
        value[0] if (value := s0.get(state_name)) is not None else
        np.full(shape, s1[state_name]), pre_shape, post_shape,
      )) for state_name, (var, reshape) in Convolution2D.var_dict.items()]
    }
