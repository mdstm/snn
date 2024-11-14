from decimal import Decimal
from typing import Union
import numpy as np

from .monitor import Monitor
from .neuron_group import NeuronGroup


class SpikeMonitor(Monitor):
    def __init__(self, target_element: NeuronGroup, sampling_period: Union[float, Decimal], position: np.ndarray):
        """
        构建SpikeMonitor实例
        """
        super().__init__(target_element, sampling_period, position)
        # self.target_element = target_element
        # self.sampling_period = sampling_period
        # self.position = position
