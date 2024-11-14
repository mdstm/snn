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
      # if
      'ConstantDecay': np.int16,
      'Vth': np.int16,
      # lif
      'tauM': float,
      'Vth': np.int16,
      'Vreset': np.int16,
      # clif
      'tauP': float,
      'tauQ': float,
      'tauM': float,
      'Vth': np.int16,
    },
    'backend': { # 需要在后端修改的变量
      # if, lif
      'Isyn': np.ndarray,
      'V': np.ndarray,
      # clif
      'Isyn': np.ndarray,
      'M': np.ndarray,
      'S': np.ndarray,
      'E': np.ndarray,
    },
  },
  'con': {
    'type': 'Connection',
    'pre': str,
    'post': str,
    'param': {
      'link_type': str,
      'max_delay': float,
      'weight': np.ndarray,
      # full
      'syn_type': ['flatten', 'basic'] | None, # 若前形状维数大于 1 则设为前者
      # conv
      'in_channels': int, # 由于 spaic bug 需要显式指定
      'kernel_size': tuple[int, int],
      'padding': tuple[int, int],
      'stride': tuple[int, int],
    },
  },
  'dec': {
    'type': 'Decoder',
    'target': str,
    'param': {
      'num': int,
      'coding_method': str,
      'coding_var_name': str,
    },
  },
  'mon': {
    'type': 'StateMonitor',
    'target': str,
    'param': {
      'var_name': str,
      'index': 'full' | tuple, # reshape 后用 np.where 直接获取元组
    },
  },
}
