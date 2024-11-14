from abc import abstractmethod
from typing import Any, Mapping
from .element_model_parameter import ElementModelParameter
from .element_model import ElementModel

class DecoderModel(ElementModelParameter, ElementModel):

    def __init__(self, parameter : Mapping[str, Any]):
        """
        构造 DecoderModel 实例。
        
        Args:
            parameter (Mapping[str, Any]): 模型参数

        """
        ElementModelParameter.__init__(self)  
        ElementModel.__init__(self) 

        for key, value in parameter.items():
            self.set_parameter(str(key), value)