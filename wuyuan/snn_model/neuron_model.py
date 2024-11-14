from abc import abstractmethod
from typing import Any
from collections.abc import Mapping
from .element_model import ElementModel
from .element_model_parameter import ElementModelParameter
from .element_model_state import ElementModelState


class NeuronModel(ElementModel, ElementModelParameter, ElementModelState):
    def __init__(self, parameter: Mapping[str, Any], initial_state: Mapping[str, Any]):
        """
        初始化 NeuronModel 实例。

        Args:
            parameter (Mapping[str, Any]): 参数的映射字典。
            initial_state (Mapping[str, Any]): 初始状态的映射字典。
        """
        ElementModel.__init__(self)
        ElementModelParameter.__init__(self)
        ElementModelState.__init__(self)

        for name, value in parameter.items():
            self.set_parameter(name, value)

        for name, value in initial_state.items():
            self.set_initial_state_value(name, value)

    @abstractmethod
    def get_input_state(self) -> str:
        """
        获取输入状态的名称。

        Returns:
            str: 输入状态的名称。
        """
        raise NotImplementedError()

