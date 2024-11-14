import spaic

from .extracter import Extracter, exts
from .util import update


def get_enc_info(a: spaic.Encoder) -> dict:
  info = {
    'type': 'Encoder',
    'param': {
      'shape': list(a.shape[1:]), # spaic 会给编码器形状添加一个维度
    },
    'model_type': 'unknown',
    'model_param': {
      'parameter': {},
    },
  }
  return update(info, exts[a.__class__.__name__](a))


class PoissonEncoding(Extracter):
  def get_info(a: spaic.Encoder) -> dict:
    return {
      'model_type': 'Poisson',
      'model_param': {
        'parameter': {
          'spike_rate_per_unit': float(a.unit_conversion),
          'time_step': float(a.dt),
        },
      },
    }
