from abc import abstractmethod
from decimal import Decimal
from typing import Union
from .observer import Observer
from .decoder_model import DecoderModel
from .element_stateful import ElementStateful
import numpy

class Decoder(Observer):

    def __init__(self, 
                 decoder_model : DecoderModel, 
                 target_element : ElementStateful, 
                 sampling_period : Union[float, Decimal], 
                 position : numpy.ndarray):
        """
        构造 Decoder 实例。

        Args:
            decoder_model (DecoderModel): 解码器模型对象。
            target_element (ElementStateful): 目标元素状态对象。
            sampling_period (Union[float, Decimal]): 采样周期，可以是浮点数或 Decimal 类型。
            position (numpy.ndarray): 位置的 NumPy 数组。

        """
        if not isinstance(decoder_model, DecoderModel):
            raise TypeError("decoder_model must be an instance of DecoderModel")
        
        super().__init__(target_element, sampling_period, position)
        self.__decoder_model = decoder_model

    def get_decoder_model(self) -> DecoderModel:
        """
        获取解码器模型。

        Returns:
            DecoderModel: 解码器模型对象。
            
        """
        return self.__decoder_model