import wuyuan.snn_model as wym

from .extracter import Extracter, exts
from .util import update


def get_enc_info(a: wym.Encoder) -> dict:
  info = {
    'type': 'Encoder',
    'param': {
      'shape': a.get_shape(),
    },
  }
  model = a.get_encoder_model()
  return update(info, exts[model.__class__.__name__](model))


class Poisson(Extracter):
  def get_info(model: wym.EncoderModel) -> dict:
    p = model._ElementModelParameter__parameter
    return {
      'param': {
        'coding_method': 'poisson',
        'dt': p['time_step'],
        'unit_conversion': p['spike_rate_per_unit'],
      },
    }
