from decimal import Decimal
from typing import Union
import numpy as np

from .element_stateful import ElementStateful
from .monitor import Monitor


class StateMonitor(Monitor):
    def __init__(self, target_element: ElementStateful, sampling_period: Union[float, Decimal], position: np.ndarray, state_name: str):
        """
        构造StateMonitor实例
        """
        super().__init__(target_element, sampling_period, position)
        self.__state_name = str(state_name)

    def get_state_name(self) -> str:
        """
        返回state_name
        Returns:
            __state_name
        """
        return self.__state_name
