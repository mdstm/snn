from decimal import Decimal
from typing import Union

import numpy as np

from .state_monitor import StateMonitor
from .connection_group import ConnectionGroup


class StateMonitorSynapse(StateMonitor):
    def __init__(self, target_element: ConnectionGroup, sampling_period: Union[float, Decimal], position: np.ndarray,
                 state_name: str):
        """
        构造StateMonitorSynapse实体
        """
        super().__init__(target_element, sampling_period, position, state_name)
