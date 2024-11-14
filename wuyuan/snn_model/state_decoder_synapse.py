from decimal import Decimal
from typing import Union
from .decoder_model import DecoderModel
from .connection_group import ConnectionGroup
from .state_decoder import StateDecoder
import numpy

class StateDecoderSynapse(StateDecoder):

    """
    突触状态解码器。
    """

    def __init__(self, 
                 decoder_model: DecoderModel, 
                 target_element: ConnectionGroup,
                 sampling_period: Union[float, Decimal], 
                 position: numpy.ndarray, 
                 state_name: str):
        """
        构造 Decoder 实例。

        Args:
            decoder_model (DecoderModel): 解码器模型对象。
            target_element (ConnectionGroup): 目标元素状态对象。
            sampling_period (Union[float, Decimal]): 采样周期，可以是浮点数或 Decimal 类型。
            position (numpy.ndarray): 位置的 NumPy 数组。
            state_name (str): 状态名称。

        """
        super().__init__(decoder_model, target_element, sampling_period, position, state_name)