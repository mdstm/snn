'''物源需要的参数'''

import numpy as np


{
  'enc': {
    'type': 'Encoder',
    'param': {
      'shape': list[int],
    },
    'model_type': str,
    'model_param': {
      'parameter': dict,
    },
  },
  'neg': {
    'type': 'NeuronGroup',
    'param': {
      'shape': list[int],
      'initial_state_value': dict[str, tuple[np.ndarray, bool]],
    },
    'model_type': str,
    'model_param': {
      'parameter': dict,
      'initial_state': dict,
    },
  },
  'con': {
    'type': 'ConnectionGroup',
    'pre': str,
    'post': str,
    'param': {
      'initial_synapse_state_value': dict[str, tuple[np.ndarray, bool]],
      'delay': float,
    },
    't_model_type': str,
    't_model_param': {
      'presynaptic_shape': list[int],
      'postsynaptic_shape': list[int],
      'parameter': dict,
    },
    's_model_type': str,
    's_model_param': {
      'parameter': dict,
      'initial_state': dict,
    },
  },
  'spdec': {
    'type': 'SpikeDecoder',
    'target': str,
    'param': {
      'sampling_period': float,
      'position': np.ndarray,
    },
    'model_type': str,
    'model_param': {
      'parameter': dict,
    },
  },
  'stdec': {
    'type': 'StateDecoderNeuron | StateDecoderSynapse',
    'target': str,
    'param': {
      'sampling_period': float,
      'position': np.ndarray,
      'state_name': str,
    },
    'model_type': str,
    'model_param': {
      'parameter': dict,
    },
  },
  'spmon': {
    'type': 'SpikeMonitor',
    'target': str,
    'param': {
      'sampling_period': float,
      'position': np.ndarray,
    },
  },
  'stmon': {
    'type': 'StateMonitorNeuron | StateMonitorSynapse',
    'target': str,
    'param': {
      'sampling_period': float,
      'position': np.ndarray,
      'state_name': str,
    },
  },
}
