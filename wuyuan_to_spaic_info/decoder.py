import math

import wuyuan.snn_model as wym

from .extracter import Extracter, exts, vars
from .util import update


def get_dec_info(a: wym.SpikeDecoder | wym.StateDecoderNeuron) -> dict:
  target = a.get_target_element()
  
  param = {
    'num': math.prod(target.get_shape()),
    'dt': float(a.get_sampling_period()),
  }
  info = {
    'type': 'Decoder',
    'target': target.get_element_id(),
    'param': param,
  }

  if isinstance(a, wym.StateDecoderNeuron):
    param['coding_var_name'], _ = vars[
      target.get_neuron_model().__class__.__name__][a.get_state_name()]

  model = a.get_decoder_model()
  return update(info, exts[model.__class__.__name__](a))


class SpikeCount(Extracter):
  def get_info(a: wym.SpikeDecoder | wym.StateDecoderNeuron) -> dict:
    return {
      'param': {
        'coding_method': 'spike_counts',
      },
    }
