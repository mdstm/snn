from typing import Sequence
from .encoder_model import EncoderModel
from .node import Node

class Encoder(Node):

    def __init__(self, encoder_model : EncoderModel, shape : Sequence[int]):
        """
        初始化 Encoder 实例。

        Args:
            encoder_model (EncoderModel): 编码器模型对象。
            shape (Sequence[int]): 编码器的形状。

        """
        if not isinstance(encoder_model, EncoderModel):
            raise TypeError("encoder_model must be an instance of EecoderModel")
        
        super().__init__(shape)
        self.__encoder_model = encoder_model
    
    # 新建
    def get_encoder_model(self) -> EncoderModel:
        return self.__encoder_model


