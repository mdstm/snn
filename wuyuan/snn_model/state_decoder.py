from decimal import Decimal
from typing import Union
from .decoder import Decoder
from .decoder import Decoder
from .decoder_model import DecoderModel
from .element_stateful import ElementStateful
import numpy

class StateDecoder(Decoder):

    def __init__(self, 
                 decoder_model : DecoderModel, 
                 target_element : ElementStateful, 
                 sampling_period : Union[float, Decimal], 
                 position : numpy.ndarray, 
                 state_name : str):
        """
        构造 StateDecoder 实例。

        Args:
            decoder_model (DecoderModel): 解码器模型对象。
            target_element (ElementStateful): 目标元素状态对象。
            sampling_period (Union[float, Decimal]): 采样周期，可以是浮点数或 Decimal 类型。
            position (numpy.ndarray): 位置的 NumPy 数组。
            state_name (str): 状态名称。

        """
        super().__init__(decoder_model, target_element, sampling_period, position)
        self.__state_name = str(state_name)


    def get_state_name(self) -> str :
        """
        获取被观测状态的名称。

        Returns:
            str: 被观测状态的名称。

        """
        return self.__state_name
