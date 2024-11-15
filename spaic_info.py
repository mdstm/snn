'''SPAIC 需要的信息'''

import numpy as np


{
  'enc': {
    'type': 'Encoder',
    'param': {
      'shape': list[int],
      'dt': float,
      'coding_method': str,
      # poisson
      'unit_conversion': float,
    },
  },
  'neg': {
    'type': 'NeuronGroup',
    'param': {
      'model': str,
      'shape': list[int],
      # lif
      'tau_m': float,
      # clif
      'tau_p': float,
      'tau_q': float,
      'tau_m': float,
    },
    'backend': [ # 需要在后端修改的变量
      # lif
      ('Vth', np.int16),
      ('Vreset', np.int16),
      ('Isyn', np.ndarray),
      ('V', np.ndarray),
      # if
      ('ConstantDecay', np.int16),
      ('Vth', np.int16),
      ('Isyn', np.ndarray),
      ('V', np.ndarray),
      # clif
      ('Vth', np.int16),
      ('Isyn', np.ndarray),
      ('M', np.ndarray),
      ('S', np.ndarray),
      ('E', np.ndarray),
    ],
  },
  'con': {
    'type': 'Connection',
    'pre': str,
    'post': str,
    'param': {
      'link_type': str,
      'max_delay': float,
      # full
      'syn_type': ['flatten', 'basic'], # 若前形状维数大于 1
      # conv
      'in_channels': int, # 由于 spaic bug 需要显式指定
      'kernel_size': tuple[int, int],
      'padding': tuple[int, int],
      'stride': tuple[int, int],
    },
    'backend': [
      ('weight', np.ndarray),
    ],
  },
  'dec': {
    'type': 'Decoder',
    'target': str,
    'param': {
      'num': int,
      'dt': float,
      'coding_var_name': str,
      'coding_method': str,
    },
  },
  'mon': {
    'type': 'StateMonitor', # spaic 一律使用 StateMonitor
    'target': str,
    'param': {
      'dt': float,
      'var_name': str,
      'index': 'full' | tuple, # reshape 后用 np.where 直接获取元组
    },
  },
}
