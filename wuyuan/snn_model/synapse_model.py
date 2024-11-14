from abc import abstractmethod
from typing import Any
from collections.abc import Mapping
from .element_model_parameter import ElementModelParameter
from .element_model import ElementModel
from .element_model_state import ElementModelState


class SynapseModel(ElementModelParameter, ElementModelState, ElementModel):
    def __init__(self, parameter: Mapping[str, Any], initial_state: Mapping[str, Any]):
        """
        构造 SynapseModel 实例。

        Args:
            parameter (Mapping[str, Any]): 模型参数
            initial_state (Mapping[str, Any]): 模型初始状态

        """
        ElementModelParameter.__init__(self)
        ElementModelState.__init__(self)
        ElementModel.__init__(self)

        for key, value in parameter.items():
            self.set_parameter(str(key), value)

        for key, value in initial_state.items():
            self.set_initial_state_value(str(key), value)

    @abstractmethod
    def get_weight_state(self) -> str:
        """
        获取作为权重的突触状态的名称。
        若突触模型没有适合作为权重的突触状态，则本方法返回 None 。

        Returns:
            str: 作为权重的突触状态的名称

        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_state(self) -> str:
        """
        获取输出状态的名称。

        Returns:
            str: 输出状态的名称

        """
        raise NotImplementedError()