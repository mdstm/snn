from decimal import Decimal
from typing import Union
from .decoder import Decoder
from .decoder_model import DecoderModel
from .neuron_group import NeuronGroup
import numpy

class SpikeDecoder(Decoder):

    def __init__(self, 
                 decoder_model : DecoderModel, 
                 target_element : NeuronGroup, 
                 sampling_period : Union[float, Decimal], 
                 position : numpy.ndarray):
        """"
        position 的元素必须为整数或布尔值。
        若 position 的元素为整数，则 position 必须为矩阵，每行为目标元素中被观测的一个位置；
        若 position 的元素为布尔值，则 position 的形状必须与被观测的目标元素相同， 
        position 中值为 True 的元素所在位置即目标元素中被观测的位置。

        Args:
            decoder_model (DecoderModel): 解码器模型对象。
            target_element (NeuronGroup): 目标元素对象。
            sampling_period (Union[float, Decimal]): 采样周期，可以是浮点数或 Decimal 类型。
            position (numpy.ndarray): 位置的 NumPy 数组。

        """
        
        super().__init__(decoder_model, target_element, sampling_period, position)

