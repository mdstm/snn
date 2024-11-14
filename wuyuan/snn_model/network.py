from decimal import Decimal
from typing import Union

from .connection_group import ConnectionGroup
from .decoder_model import DecoderModel
from .encoder import Encoder
from .encoder_model import EncoderModel
from .neuron_group import NeuronGroup
from .neuron_model import NeuronModel
from .spike_decoder import SpikeDecoder
from .spike_monitor import SpikeMonitor
from .state_decoder_neuron import StateDecoderNeuron
from .state_decoder_synapse import StateDecoderSynapse
from .state_monitor_neuron import StateMonitorNeuron
from .state_monitor_synapse import StateMonitorSynapse
from .synapse_model import SynapseModel
from .topology_model import TopologyModel
from .utils.stateless_element_map import create_stateless_element_map_class
from .utils.stateful_element_map import create_stateful_element_map_class


_NeuronModelMap = create_stateless_element_map_class(
    NeuronModel, element_type_name_id_prefix="NeuronModel",
    element_type_name_class="NeuronModel",
    element_type_name_member="neuron_model", element_type_name_doc="神经元模型")
_EncoderModelMap = create_stateless_element_map_class(
    EncoderModel, element_type_name_id_prefix="EncoderModel",
    element_type_name_class="EncoderModel",
    element_type_name_member="encoder_model",
    element_type_name_doc="脉冲编码器模型")
_SynapseModelMap = create_stateless_element_map_class(
    SynapseModel, element_type_name_id_prefix="SynapseModel",
    element_type_name_class="SynapseModel",
    element_type_name_member="synapse_model", element_type_name_doc="突触模型")
_TopologyModelMap = create_stateless_element_map_class(
    TopologyModel, element_type_name_id_prefix="TopologyModel",
    element_type_name_class="TopologyModel",
    element_type_name_member="topology_model",
    element_type_name_doc="拓扑结构模型")
_DecoderModelMap = create_stateless_element_map_class(
    DecoderModel, element_type_name_id_prefix="DecoderModel",
    element_type_name_class="DecoderModel",
    element_type_name_member="decoder_model", element_type_name_doc="解码器模型")

_NeuronGroupMap = create_stateful_element_map_class(
    NeuronGroup, element_type_name_id_prefix="NeuronGroup",
    element_type_name_class="NeuronGroup",
    element_type_name_member="neuron_group", element_type_name_doc="神经元组")
_EncoderMap = create_stateful_element_map_class(
    Encoder, element_type_name_id_prefix="Encoder",
    element_type_name_class="Encoder", element_type_name_member="encoder",
    element_type_name_doc="脉冲编码器")
_ConnectionGroupMap = create_stateful_element_map_class(
    ConnectionGroup, element_type_name_id_prefix="ConnectionGroup",
    element_type_name_class="ConnectionGroup",
    element_type_name_member="connection_group",
    element_type_name_doc="连接组")
_SpikeMonitorMap = create_stateful_element_map_class(
    SpikeMonitor, element_type_name_id_prefix="SpikeMonitor",
    element_type_name_class="SpikeMonitor",
    element_type_name_member="spike_monitor",
    element_type_name_doc="脉冲监视器")
_StateMonitorNeuronMap = create_stateful_element_map_class(
    StateMonitorNeuron, element_type_name_id_prefix="StateMonitorNeuron",
    element_type_name_class="StateMonitorNeuron",
    element_type_name_member="state_monitor_neuron",
    element_type_name_doc="神经元状态监视器")
_StateMonitorSynapseMap = create_stateful_element_map_class(
    StateMonitorSynapse, element_type_name_id_prefix="StateMonitorSynapse",
    element_type_name_class="StateMonitorSynapse",
    element_type_name_member="state_monitor_synapse",
    element_type_name_doc="突触状态监视器")
_SpikeDecoderMap = create_stateful_element_map_class(
    SpikeDecoder, element_type_name_id_prefix="SpikeDecoder",
    element_type_name_class="SpikeDecoder",
    element_type_name_member="spike_decoder",
    element_type_name_doc="脉冲解码器")
_StateDecoderNeuronMap = create_stateful_element_map_class(
    StateDecoderNeuron, element_type_name_id_prefix="StateDecoderNeuron",
    element_type_name_class="StateDecoderNeuron",
    element_type_name_member="state_decoder_neuron",
    element_type_name_doc="神经元状态解码器")
_StateDecoderSynapseMap = create_stateful_element_map_class(
    StateDecoderSynapse, element_type_name_id_prefix="StateDecoderSynapse",
    element_type_name_class="StateDecoderSynapse",
    element_type_name_member="state_decoder_synapse",
    element_type_name_doc="突触状态解码器")


class Network(
        _NeuronModelMap, _EncoderModelMap, _SynapseModelMap, _TopologyModelMap,
        _DecoderModelMap, _NeuronGroupMap, _EncoderMap, _ConnectionGroupMap,
        _SpikeMonitorMap, _StateMonitorNeuronMap, _StateMonitorSynapseMap,
        _SpikeDecoderMap, _StateDecoderNeuronMap, _StateDecoderSynapseMap):
    """SNN模型。"""
    def __init__(self):
        _NeuronModelMap.__init__(self)
        _EncoderModelMap.__init__(self)
        _SynapseModelMap.__init__(self)
        _TopologyModelMap.__init__(self)
        _DecoderModelMap.__init__(self)
        _NeuronGroupMap.__init__(self)
        _EncoderMap.__init__(self)
        _ConnectionGroupMap.__init__(self)
        _SpikeMonitorMap.__init__(self)
        _StateMonitorNeuronMap.__init__(self)
        _StateMonitorSynapseMap.__init__(self)
        _SpikeDecoderMap.__init__(self)
        _StateDecoderNeuronMap.__init__(self)
        _StateDecoderSynapseMap.__init__(self)
        self.__dependency_neuron_model: dict[str, set[str]] = dict()
        self.__dependency_encoder_model: dict[str, set[str]] = dict()
        self.__dependency_synapse_model: dict[str, set[str]] = dict()
        self.__dependency_topology_model: dict[str, set[str]] = dict()
        self.__dependency_decoder_model_spike_decoder: dict[str, set[str]] = \
            dict()
        self.__dependency_decoder_model_state_decoder_neuron: \
            dict[str, set[str]] = \
            dict()
        self.__dependency_decoder_model_state_decoder_synapse: \
            dict[str, set[str]] = \
            dict()
        self.__dependency_neuron_group_connection_group: \
            dict[str, set[str]] = \
            dict()
        self.__dependency_neuron_group_spike_monitor: dict[str, set[str]] = \
            dict()
        self.__dependency_neuron_group_state_monitor: dict[str, set[str]] = \
            dict()
        self.__dependency_neuron_group_spike_decoder: dict[str, set[str]] = \
            dict()
        self.__dependency_neuron_group_state_decoder: dict[str, set[str]] = \
            dict()
        self.__dependency_encoder: dict[str, set[str]] = dict()
        self.__dependency_connection_group_state_monitor: \
            dict[str, set[str]] = \
            dict()
        self.__dependency_connection_group_state_decoder: \
            dict[str, set[str]] = \
            dict()
        self.__time_step: Union[float, Decimal] = 1.0
        self.__time_window: Union[float, Decimal] = 0.0

    def neuron_model_remove(self, id_: str) -> NeuronModel:
        neuron_model = _NeuronModelMap.neuron_model_remove(self, id_)
        dependent_neuron_group = self.__dependency_neuron_model.pop(
            str(id_), [])
        for neuron_group_id in dependent_neuron_group:
            self.neuron_group_remove(neuron_group_id)
        return neuron_model

    neuron_model_remove.__doc__ = \
        _NeuronModelMap.neuron_model_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除依赖被移除神经元模型的神经元组。
        """

    def encoder_model_remove(self, id_: str) -> EncoderModel:
        encoder_model = _EncoderModelMap.encoder_model_remove(self, id_)
        dependent_encoder = self.__dependency_encoder_model.pop(str(id_), [])
        for encoder_id in dependent_encoder:
            self.encoder_remove(encoder_id)
        return encoder_model

    encoder_model_remove.__doc__ = \
        _EncoderModelMap.encoder_model_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除依赖被移除脉冲编码器模型的脉冲编码器。
        """

    def synapse_model_remove(self, id_: str) -> SynapseModel:
        synapse_model = _SynapseModelMap.synapse_model_remove(self, id_)
        dependent_connection_group = self.__dependency_synapse_model.pop(
            str(id_), [])
        for connection_group_id in dependent_connection_group:
            self.connection_group_remove(connection_group_id)
        return synapse_model

    synapse_model_remove.__doc__ = \
        _SynapseModelMap.synapse_model_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除依赖被移除突触模型的连接组。
        """

    def topology_model_remove(self, id_: str) -> TopologyModel:
        topology_model = _TopologyModelMap.topology_model_remove(self, id_)
        dependent_connection_group = self.__dependency_topology_model.pop(
            str(id_), [])
        for connection_group_id in dependent_connection_group:
            self.connection_group_remove(connection_group_id)
        return topology_model

    topology_model_remove.__doc__ = \
        _TopologyModelMap.topology_model_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除依赖被移除拓扑结构模型的连接组。
        """

    def decoder_model_remove(self, id_: str) -> DecoderModel:
        id__ = str(id_)
        decoder_model = _DecoderModelMap.decoder_model_remove(self, id__)
        dependent_spike_decoder = \
            self.__dependency_decoder_model_spike_decoder.pop(id__, [])
        for spike_decoder_id in dependent_spike_decoder:
            self.spike_decoder_remove(spike_decoder_id)
        dependent_state_decoder_neuron = \
            self.__dependency_decoder_model_state_decoder_neuron.pop(id__, [])
        for state_decoder_neuron_id in dependent_state_decoder_neuron:
            self.state_decoder_neuron_remove(state_decoder_neuron_id)
        dependent_state_decoder_synapse = \
            self.__dependency_decoder_model_state_decoder_synapse.pop(id__, [])
        for state_decoder_synapse_id in dependent_state_decoder_synapse:
            self.state_decoder_synapse_remove(state_decoder_synapse_id)
        return decoder_model

    decoder_model_remove.__doc__ = \
        _DecoderModelMap.decoder_model_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除依赖被移除解码器模型的解码器。
        """

    def neuron_group_add(self, neuron_group: NeuronGroup, id_: str = None) \
            -> NeuronGroup:
        if not isinstance(neuron_group, NeuronGroup):
            raise TypeError("neuron_group must be a NeuronGroup instance.")
        neuron_model = neuron_group.get_neuron_model()
        if not _NeuronModelMap.neuron_model_contains_instance(
                self, neuron_model):
            raise ValueError(
                "neuron_group must use a NeuronModel instance that is "
                "already added to the SNN model.")
        neuron_model_id = neuron_model.get_element_id()
        neuron_group_ = None
        try:
            neuron_group_ = _NeuronGroupMap.neuron_group_add(
                self, neuron_group, id_)
            self.__dependency_add(
                self.__dependency_neuron_model, neuron_model_id,
                neuron_group_.get_element_id())
        except:
            if neuron_group_ is not None:
                neuron_group_id = neuron_group_.get_element_id()
                _NeuronGroupMap.neuron_group_remove(self, neuron_group_id)
                self.__dependency_remove(
                    self.__dependency_neuron_model, neuron_model_id,
                    neuron_group_id)
            raise
        return neuron_group_

    neuron_group_add.__doc__ = _NeuronGroupMap.neuron_group_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加神经元组对神经元模型的依赖关系。

            2. 若 `neuron_group` 的神经元模型不在SNN模型中，即，神经元模型的
            NeuronModel实例不在SNN模型中，则本方法抛出ValueError。
        """

    def neuron_group_remove(self, id_: str) -> NeuronGroup:
        neuron_group = _NeuronGroupMap.neuron_group_remove(self, str(id_))
        if neuron_group is None:
            return None
        id__ = neuron_group.get_element_id()
        self.__dependency_remove(
            self.__dependency_neuron_model,
            neuron_group.get_neuron_model().get_element_id(), id__)
        dependent_connection_group = \
            self.__dependency_neuron_group_connection_group.pop(id__, [])
        for connection_group_id in dependent_connection_group:
            self.connection_group_remove(connection_group_id)
        dependent_spike_monitor = \
            self.__dependency_neuron_group_spike_monitor.pop(id__, [])
        for spike_monitor_id in dependent_spike_monitor:
            self.spike_monitor_remove(spike_monitor_id)
        dependent_state_monitor = \
            self.__dependency_neuron_group_state_monitor.pop(id__, [])
        for state_monitor_id in dependent_state_monitor:
            self.state_monitor_neuron_remove(state_monitor_id)
        dependent_spike_decoder = \
            self.__dependency_neuron_group_spike_decoder.pop(id__, [])
        for spike_decoder_id in dependent_spike_decoder:
            self.spike_decoder_remove(spike_decoder_id)
        dependent_state_decoder = \
            self.__dependency_neuron_group_state_decoder.pop(id__, [])
        for state_decoder_id in dependent_state_decoder:
            self.state_decoder_neuron_remove(state_decoder_id)
        return neuron_group

    neuron_group_remove.__doc__ = \
        _NeuronGroupMap.neuron_group_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除神经元组对神经元模型的依赖关系，移除依赖被移除神经元
            组的连接组和观测器。
        """

    def encoder_add(self, encoder: Encoder, id_: str = None) -> Encoder:
        if not isinstance(encoder, Encoder):
            raise TypeError("encoder must be an Encoder instance.")
        encoder_model = encoder.get_encoder_model()
        if not _EncoderModelMap.encoder_model_contains_instance(
                self, encoder_model):
            raise ValueError(
                "encoder must use an EncoderModel instance that is already "
                "added to the SNN model.")
        encoder_model_id = encoder_model.get_element_id()
        encoder_ = None
        try:
            encoder_ = _EncoderMap.encoder_add(self, encoder, id_)
            self.__dependency_add(
                self.__dependency_encoder_model, encoder_model_id,
                encoder_.get_element_id())
        except:
            if encoder_ is not None:
                encoder_id = encoder_.get_element_id()
                _EncoderMap.encoder_remove(self, encoder_id)
                self.__dependency_remove(
                    self.__dependency_encoder_model, encoder_model_id,
                    encoder_id)
            raise
        return encoder_

    encoder_add.__doc__ = _EncoderMap.encoder_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加脉冲编码器对脉冲编码器模型的依赖关系。

            2. 若 `encoder` 的脉冲编码器模型不在SNN模型中，即，脉冲编码器模型的
            EncoderModel实例不在SNN模型中，则本方法抛出ValueError。
        """

    def encoder_remove(self, id_: str) -> Encoder:
        encoder = _EncoderMap.encoder_remove(self, str(id_))
        if encoder is None:
            return None
        id__ = encoder.get_element_id()
        self.__dependency_remove(
            self.__dependency_encoder_model,
            encoder.get_encoder_model().get_element_id(), id__)
        dependent_connection_group = self.__dependency_encoder.pop(id__, [])
        for connection_group_id in dependent_connection_group:
            self.connection_group_remove(connection_group_id)
        return encoder

    encoder_remove.__doc__ = _EncoderMap.encoder_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除脉冲编码器对脉冲编码器模型的依赖关系，移除依赖被移除
            脉冲编码器的连接组。
        """

    def connection_group_add(
            self, connection_group: ConnectionGroup, id_: str = None) \
            -> ConnectionGroup:
        if not isinstance(connection_group, ConnectionGroup):
            raise TypeError(
                "connection_group must be a ConnectionGroup instance.")
        synapse_model = connection_group.get_synapse_model()
        if not _SynapseModelMap.synapse_model_contains_instance(
                self, synapse_model):
            raise ValueError(
                "connection_group must use a SynapseModel instance that is "
                "already added to the SNN model.")
        topology_model = connection_group.get_topology_model()
        if not _TopologyModelMap.topology_model_contains_instance(
                self, topology_model):
            raise ValueError(
                "connection_group must use a TopologyModel instance that is "
                "already added to the SNN model.")
        presynaptic_node = connection_group.get_presynaptic_node()
        if isinstance(presynaptic_node, NeuronGroup):
            if not _NeuronGroupMap.neuron_group_contains_instance(
                    self, presynaptic_node):
                raise ValueError(
                    "connection_group must use a presynaptic NeuronGroup "
                    "instance that is already added to the SNN model.")
        elif isinstance(presynaptic_node, Encoder):
            if not _EncoderMap.encoder_contains_instance(
                    self, presynaptic_node):
                raise ValueError(
                    "connection_group must use a presynaptic Encoder instance "
                    "that is already added to the SNN model.")
        else:
            raise TypeError("The type of the presynaptic node is unknown.")
        postsynaptic_node = connection_group.get_postsynaptic_node()
        if not isinstance(postsynaptic_node, NeuronGroup):
            raise TypeError(
                "The postsynaptic node of connection_group must be a "
                "NeuronGroup instance.")
        if not _NeuronGroupMap.neuron_group_contains_instance(
                self, postsynaptic_node):
            raise ValueError(
                "connection_group must use a postsynaptic NeuronGroup "
                "instance that is already added to the SNN model.")
        synapse_model_id = synapse_model.get_element_id()
        topology_model_id = topology_model.get_element_id()
        presynaptic_node_id = presynaptic_node.get_element_id()
        postsynaptic_node_id = postsynaptic_node.get_element_id()
        connection_group_ = None
        try:
            connection_group_ = _ConnectionGroupMap.connection_group_add(
                self, connection_group, id_)
            connection_group_id = connection_group_.get_element_id()
            self.__dependency_add(
                self.__dependency_synapse_model, synapse_model_id,
                connection_group_id)
            self.__dependency_add(
                self.__dependency_topology_model, topology_model_id,
                connection_group_id)
            if isinstance(presynaptic_node, NeuronGroup):
                self.__dependency_add(
                    self.__dependency_neuron_group_connection_group,
                    presynaptic_node_id, connection_group_id)
            elif isinstance(presynaptic_node, Encoder):
                self.__dependency_add(
                    self.__dependency_encoder, presynaptic_node_id,
                    connection_group_id)
            else:
                raise TypeError("The type of the presynaptic node is unknown.")
            self.__dependency_add(
                self.__dependency_neuron_group_connection_group,
                postsynaptic_node_id, connection_group_id)
        except:
            if connection_group_ is not None:
                connection_group_id = connection_group_.get_element_id()
                _ConnectionGroupMap.connection_group_remove(
                    self, connection_group_id)
                self.__dependency_remove(
                    self.__dependency_synapse_model, synapse_model_id,
                    connection_group_id)
                self.__dependency_remove(
                    self.__dependency_topology_model, topology_model_id,
                    connection_group_id)
                if isinstance(presynaptic_node, NeuronGroup):
                    self.__dependency_remove(
                        self.__dependency_neuron_group_connection_group,
                        presynaptic_node_id, connection_group_id)
                elif isinstance(presynaptic_node, Encoder):
                    self.__dependency_remove(
                        self.__dependency_encoder, presynaptic_node_id,
                        connection_group_id)
                else:
                    raise TypeError(
                        "The type of the presynaptic node is unknown.")
                self.__dependency_remove(
                    self.__dependency_neuron_group_connection_group,
                    postsynaptic_node_id, connection_group_id)
            raise
        return connection_group_

    connection_group_add.__doc__ = \
        _ConnectionGroupMap.connection_group_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加连接组对突触模型、拓扑结构模型和节点的依赖关系。

            2. 若 `connection_group` 的突触模型或拓扑结构模型不在SNN模型中，即，突触模型
            的SynapseModel实例或拓扑结构模型的TopologyModel实例不在SNN模型中，则本方法
            抛出ValueError。

            3. 若 `connection_group` 连接的节点不在SNN模型中，即，节点的Node实例不在
            SNN模型中，则本方法抛出ValueError。
        """

    def connection_group_remove(self, id_: str) -> ConnectionGroup:
        connection_group = _ConnectionGroupMap.connection_group_remove(
            self, str(id_))
        if connection_group is None:
            return None
        id__ = connection_group.get_element_id()
        self.__dependency_remove(
            self.__dependency_synapse_model,
            connection_group.get_synapse_model().get_element_id(), id__)
        self.__dependency_remove(
            self.__dependency_topology_model,
            connection_group.get_topology_model().get_element_id(), id__)
        presynaptic_node = connection_group.get_presynaptic_node()
        presynaptic_node_id = presynaptic_node.get_element_id()
        if isinstance(presynaptic_node, NeuronGroup):
            self.__dependency_remove(
                self.__dependency_neuron_group_connection_group,
                presynaptic_node_id, id__)
        elif isinstance(presynaptic_node, Encoder):
            self.__dependency_remove(
                self.__dependency_encoder, presynaptic_node_id, id__)
        else:
            raise TypeError("The type of the presynaptic node is unknown.")
        self.__dependency_remove(
            self.__dependency_neuron_group_connection_group,
            connection_group.get_postsynaptic_node().get_element_id(), id__)
        dependent_state_monitor = \
            self.__dependency_connection_group_state_monitor.pop(id__, [])
        for state_monitor_id in dependent_state_monitor:
            self.state_monitor_synapse_remove(state_monitor_id)
        dependent_state_decoder = \
            self.__dependency_connection_group_state_decoder.pop(id__, [])
        for state_decoder_id in dependent_state_decoder:
            self.state_decoder_synapse_remove(state_decoder_id)
        return connection_group

    connection_group_remove.__doc__ = \
        _ConnectionGroupMap.connection_group_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除连接组对突触模型、拓扑结构模型和节点的依赖关系，移除
            依赖被移除连接组的观测器。
        """

    def spike_monitor_add(self, spike_monitor: SpikeMonitor, id_: str = None) \
            -> SpikeMonitor:
        if not isinstance(spike_monitor, SpikeMonitor):
            raise TypeError("spike_monitor must be a SpikeMonitor instance.")
        target = spike_monitor.get_target_element()
        if not _NeuronGroupMap.neuron_group_contains_instance(self, target):
            raise ValueError(
                "spike_monitor must target a NeuronGroup instance that is "
                "already added to the SNN model.")
        target_id = target.get_element_id()
        spike_monitor_ = None
        try:
            spike_monitor_ = _SpikeMonitorMap.spike_monitor_add(
                self, spike_monitor, id_)
            self.__dependency_add(
                self.__dependency_neuron_group_spike_monitor, target_id,
                spike_monitor_.get_element_id())
        except:
            if spike_monitor_ is not None:
                spike_monitor_id = spike_monitor_.get_element_id()
                _SpikeMonitorMap.spike_monitor_remove(self, spike_monitor_id)
                self.__dependency_remove(
                    self.__dependency_neuron_group_spike_monitor, target_id,
                    spike_monitor_id)
            raise
        return spike_monitor_

    spike_monitor_add.__doc__ = _SpikeMonitorMap.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加脉冲监视器对目标神经元组的依赖关系。

            2. 若 `spike_monitor` 的目标神经元组不在SNN模型中，即，神经元组的
            NeuronGroup实例不在SNN模型中，则本方法抛出ValueError。
        """

    def spike_monitor_remove(self, id_: str) -> SpikeMonitor:
        spike_monitor = _SpikeMonitorMap.spike_monitor_remove(self, id_)
        if spike_monitor is None:
            return None
        self.__dependency_remove(
            self.__dependency_neuron_group_spike_monitor,
            spike_monitor.get_target_element().get_element_id(),
            spike_monitor.get_element_id())
        return spike_monitor

    spike_monitor_remove.__doc__ = \
        _SpikeMonitorMap.spike_monitor_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除脉冲监视器对目标神经元组的依赖关系。
        """

    def state_monitor_neuron_add(
            self, state_monitor_neuron: StateMonitorNeuron, id_: str = None) \
            -> StateMonitorNeuron:
        if not isinstance(state_monitor_neuron, StateMonitorNeuron):
            raise TypeError(
                "state_monitor_neuron must be a StateMonitorNeuron instance.")
        target = state_monitor_neuron.get_target_element()
        if not _NeuronGroupMap.neuron_group_contains_instance(self, target): # 修改
            raise ValueError(
                "state_monitor_neuron must target a NeuronGroup instance that "
                "is already added to the SNN model.")
        target_id = target.get_element_id()
        state_monitor_neuron_ = None
        try:
            state_monitor_neuron_ = \
                _StateMonitorNeuronMap.state_monitor_neuron_add(
                    self, state_monitor_neuron, id_)
            self.__dependency_add(
                self.__dependency_neuron_group_state_monitor, target_id,
                state_monitor_neuron_.get_element_id())
        except:
            if state_monitor_neuron_ is not None:
                state_monitor_neuron_id = \
                    state_monitor_neuron_.get_element_id()
                _StateMonitorNeuronMap.state_monitor_neuron_remove(
                    self, state_monitor_neuron_id)
                self.__dependency_remove(
                    self.__dependency_neuron_group_state_monitor, target_id,
                    state_monitor_neuron_id)
            raise
        return state_monitor_neuron_

    state_monitor_neuron_add.__doc__ = \
        _StateMonitorNeuronMap.state_monitor_neuron_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加神经元状态监视器对目标神经元组的依赖关系。

            2. 若 `state_monitor_neuron` 的目标神经元组不在SNN模型中，即，神经元组的
            NeuronGroup实例不在SNN模型中，则本方法抛出ValueError。
        """

    def state_monitor_neuron_remove(self, id_: str) -> StateMonitorNeuron:
        state_monitor_neuron = \
            _StateMonitorNeuronMap.state_monitor_neuron_remove(self, id_)
        if state_monitor_neuron is None:
            return None
        self.__dependency_remove(
            self.__dependency_neuron_group_state_monitor,
            state_monitor_neuron.get_target_element().get_element_id(),
            state_monitor_neuron.get_element_id())
        return state_monitor_neuron

    state_monitor_neuron_remove.__doc__ = \
        _StateMonitorNeuronMap.state_monitor_neuron_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除神经元状态监视器对目标神经元组的依赖关系。
        """

    def state_monitor_synapse_add(
            self, state_monitor_synapse: StateMonitorSynapse,
            id_: str = None) \
            -> StateMonitorSynapse:
        if not isinstance(state_monitor_synapse, StateMonitorSynapse):
            raise TypeError(
                "state_monitor_synapse must be a StateMonitorSynapse "
                "instance.")
        target = state_monitor_synapse.get_target_element()
        if not _ConnectionGroupMap.connection_group_contains_instance(
                self, target):
            raise ValueError(
                "state_monitor_synapse must target a ConnectionGroup instance "
                "that is already added to the SNN model.")
        target_id = target.get_element_id()
        state_monitor_synapse_ = None
        try:
            state_monitor_synapse_ = \
                _StateMonitorSynapseMap.state_monitor_synapse_add(
                    self, state_monitor_synapse, id_)
            self.__dependency_add(
                self.__dependency_connection_group_state_monitor, target_id,
                state_monitor_synapse_.get_element_id())
        except:
            if state_monitor_synapse_ is not None:
                state_monitor_synapse_id = \
                    state_monitor_synapse_.get_element_id()
                _StateMonitorSynapseMap.state_monitor_synapse_remove(
                    self, state_monitor_synapse_id)
                self.__dependency_remove(
                    self.__dependency_connection_group_state_monitor,
                    target_id, state_monitor_synapse_id)
            raise
        return state_monitor_synapse_

    state_monitor_synapse_add.__doc__ = \
        _StateMonitorSynapseMap.state_monitor_synapse_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加突触状态监视器对目标连接组的依赖关系。

            2. 若 `state_monitor_synapse` 的目标连接组不在SNN模型中，即，连接组的
            ConnectionGroup实例不在SNN模型中，则本方法抛出ValueError。
        """

    def state_monitor_synapse_remove(self, id_: str) -> StateMonitorSynapse:
        state_monitor_synapse = \
            _StateMonitorSynapseMap.state_monitor_synapse_remove(self, id_)
        if state_monitor_synapse is None:
            return None
        self.__dependency_remove(
            self.__dependency_connection_group_state_monitor,
            state_monitor_synapse.get_target_element().get_element_id(),
            state_monitor_synapse.get_element_id())
        return state_monitor_synapse

    state_monitor_synapse_remove.__doc__ = \
        _StateMonitorSynapseMap.state_monitor_synapse_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除突触状态监视器对目标连接组的依赖关系。
        """

    def spike_decoder_add(self, spike_decoder: SpikeDecoder, id_: str = None) \
            -> SpikeDecoder:
        if not isinstance(spike_decoder, SpikeDecoder):
            raise TypeError("spike_decoder must be a SpikeDecoder instance.")
        decoder_model = spike_decoder.get_decoder_model()
        if not _DecoderModelMap.decoder_model_contains_instance(
                self, decoder_model):
            raise ValueError(
                "spike_decoder must use a DecoderModel that is already added "
                "to the SNN model.")
        target = spike_decoder.get_target_element()
        if not _NeuronGroupMap.neuron_group_contains_instance(self, target):
            raise ValueError(
                "spike_decoder must target a NeuronGroup instance that is "
                "already added to the SNN model.")
        decoder_model_id = decoder_model.get_element_id()
        target_id = target.get_element_id()
        spike_decoder_ = None
        try:
            spike_decoder_ = _SpikeDecoderMap.spike_decoder_add(
                self, spike_decoder, id_)
            spike_decoder_id = spike_decoder_.get_element_id()
            self.__dependency_add(
                self.__dependency_decoder_model_spike_decoder,
                decoder_model_id, spike_decoder_id)
            self.__dependency_add(
                self.__dependency_neuron_group_spike_decoder, target_id,
                spike_decoder_id)
        except:
            if spike_decoder_ is not None:
                spike_decoder_id = spike_decoder_.get_element_id()
                _SpikeDecoderMap.spike_decoder_remove(self, spike_decoder_id)
                self.__dependency_remove(
                    self.__dependency_decoder_model_spike_decoder,
                    decoder_model_id, spike_decoder_id)
                self.__dependency_remove(
                    self.__dependency_neuron_group_spike_decoder, target_id,
                    spike_decoder_id)
            raise
        return spike_decoder_

    spike_decoder_add.__doc__ = _SpikeDecoderMap.spike_decoder_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加脉冲解码器对解码器模型和目标神经元组的依赖关系。

            2. 若 `spike_decoder` 的解码器模型不在SNN模型中，即，解码器模型的
            DecoderModel实例不在SNN模型中，则本方法抛出ValueError。

            3. 若 `spike_decoder` 的目标神经元组不在SNN模型中，即，神经元组的
            NeuronGroup实例不在SNN模型中，则本方法抛出ValueError。
        """

    def spike_decoder_remove(self, id_: str) -> SpikeDecoder:
        spike_decoder = _SpikeDecoderMap.spike_decoder_remove(id_)
        if spike_decoder is None:
            return None
        id__ = spike_decoder.get_element_id()
        self.__dependency_remove(
            self.__dependency_decoder_model_spike_decoder,
            spike_decoder.get_decoder_model().get_element_id(), id__)
        self.__dependency_remove(
            self.__dependency_neuron_group_spike_decoder,
            spike_decoder.get_target_element().get_element_id(), id__)
        return spike_decoder

    spike_decoder_remove.__doc__ = \
        _SpikeDecoderMap.spike_decoder_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除脉冲解码器对解码器模型和目标神经元组的依赖关系。
        """

    def state_decoder_neuron_add(
            self, state_decoder_neuron: StateDecoderNeuron, id_: str = None) \
            -> StateDecoderNeuron:
        if not isinstance(state_decoder_neuron, StateDecoderNeuron):
            raise TypeError(
                "state_decoder_neuron must be a StateDecoderNeuron instance.")
        decoder_model = state_decoder_neuron.get_decoder_model()
        if not _DecoderModelMap.decoder_model_contains_instance(
                self, decoder_model):
            raise ValueError(
                "state_decoder_neuron must use a DecoderModel instance that "
                "is already added to the SNN model.")
        target = state_decoder_neuron.get_target_element()
        if not _NeuronGroupMap.neuron_group_contains_instance(self, target):
            raise ValueError(
                "state_decoder_neuron must target a NeuronGroup instance that "
                "is already added to the SNN model.")
        decoder_model_id = decoder_model.get_element_id()
        target_id = target.get_element_id()
        state_decoder_neuron_ = None
        try:
            state_decoder_neuron_ = \
                _StateDecoderNeuronMap.state_decoder_neuron_add(
                    self, state_decoder_neuron, id_)
            state_decoder_neuron_id = state_decoder_neuron_.get_element_id()
            self.__dependency_add(
                self.__dependency_decoder_model_state_decoder_neuron,
                decoder_model_id, state_decoder_neuron_id)
            self.__dependency_add(
                self.__dependency_neuron_group_state_decoder, target_id,
                state_decoder_neuron_id)
        except:
            if state_decoder_neuron_ is not None:
                state_decoder_neuron_id = \
                    state_decoder_neuron_.get_element_id()
                _StateDecoderNeuronMap.state_decoder_neuron_remove(
                    self, state_decoder_neuron_id)
                self.__dependency_remove(
                    self.__dependency_decoder_model_state_decoder_neuron,
                    decoder_model_id, state_decoder_neuron_id)
                self.__dependency_remove(
                    self.__dependency_neuron_group_state_decoder, target_id,
                    state_decoder_neuron_id)
            raise
        return state_decoder_neuron_

    state_decoder_neuron_add.__doc__ = \
        _StateDecoderNeuronMap.state_decoder_neuron_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加神经元状态解码器对解码器模型和目标神经元组的依赖
            关系。

            2. 若 `state_decoder_neuron` 的解码器模型不在SNN模型中，即，解码器模型的
            DecoderModel实例不在SNN模型中，则本方法抛出ValueError。

            3. 若 `state_decoder_neuron` 的目标神经元组不在SNN模型中，即，神经元组的
            NeuronGroup实例不在SNN模型中，则本方法抛出ValueError。
        """

    def state_decoder_neuron_remove(self, id_: str) -> StateDecoderNeuron:
        state_decoder_neuron = \
            _StateDecoderNeuronMap.state_decoder_neuron_remove(id_)
        if state_decoder_neuron is None:
            return None
        id__ = state_decoder_neuron.get_element_id()
        self.__dependency_remove(
            self.__dependency_decoder_model_state_decoder_neuron,
            state_decoder_neuron.get_decoder_model.get_element_id(), id__)
        self.__dependency_remove(
            self.__dependency_neuron_group_state_decoder,
            state_decoder_neuron.get_target_element().get_element_id(), id__)
        return state_decoder_neuron

    state_decoder_neuron_remove.__doc__ = \
        _StateDecoderNeuronMap.state_decoder_neuron_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除神经元状态解码器对解码器模型和目标神经元组的依赖关
            系。
        """

    def state_decoder_synapse_add(
            self, state_decoder_synapse: StateDecoderSynapse,
            id_: str = None) \
            -> StateDecoderSynapse:
        if not isinstance(state_decoder_synapse, StateDecoderSynapse):
            raise TypeError(
                "state_decoder_synapse must be a StateDecoderSynapse "
                "instance.")
        decoder_model = state_decoder_synapse.get_decoder_model()
        if not _DecoderModelMap.decoder_model_contains_instance(
                self, decoder_model):
            raise ValueError(
                "state_decoder_synapse must use a DecoderModel instance that "
                "is already added to the SNN model.")
        target = state_decoder_synapse.get_target_element()
        if not _ConnectionGroupMap.connection_group_contains_instance(
                self, target):
            raise ValueError(
                "state_decoder_synapse must target a ConnectionGroup instance "
                "that is already added to the SNN model.")
        decoder_model_id = decoder_model.get_element_id()
        target_id = target.get_element_id()
        state_decoder_synapse_ = None
        try:
            state_decoder_synapse_ = \
                _StateDecoderSynapseMap.state_decoder_synapse_add(
                    self, state_decoder_synapse, id_)
            state_decoder_synapse_id = state_decoder_synapse_.get_element_id()
            self.__dependency_add(
                self.__dependency_decoder_model_state_decoder_synapse,
                decoder_model_id, state_decoder_synapse_id)
            self.__dependency_add(
                self.__dependency_connection_group_state_decoder, target_id,
                state_decoder_synapse_id)
        except:
            if state_decoder_synapse_ is not None:
                state_decoder_synapse_id = \
                    state_decoder_synapse_.get_element_id()
                _StateDecoderSynapseMap.state_decoder_synapse_remove(
                    self, state_decoder_synapse_id)
                self.__dependency_remove(
                    self.__dependency_decoder_model_state_decoder_synapse,
                    decoder_model_id, state_decoder_synapse_id)
                self.__dependency_remove(
                    self.__dependency_connection_group_state_decoder,
                    target_id, state_decoder_synapse_id)
            raise
        return state_decoder_synapse_

    state_decoder_synapse_add.__doc__ = \
        _StateDecoderSynapseMap.state_decoder_synapse_add.__doc__ + \
        """

        提示:
            1. 本方法扩展基类的同名方法，添加突触状态解码器对解码器模型和目标连接组的依赖关
            系。

            2. 若 `state_decoder_synapse` 的解码器模型不在SNN模型中，即，解码器模型的
            DecoderModel实例不在SNN模型中，则本方法抛出ValueError。

            3. 若 `state_decoder_synapse` 的目标连接组不在SNN模型中，即，连接组的
            ConnectionGroup实例不在SNN模型中，则本方法抛出ValueError。
        """

    def state_decoder_synapse_remove(self, id_: str) ->StateDecoderSynapse:
        state_decoder_synapse = \
            _StateDecoderSynapseMap.state_decoder_synapse_remove(self, id_)
        if state_decoder_synapse is None:
            return None
        id__ = state_decoder_synapse.get_element_id()
        self.__dependency_remove(
            self.__dependency_decoder_model_state_decoder_synapse,
            state_decoder_synapse.get_decoder_model().get_element_id(), id__)
        self.__dependency_remove(
            self.__dependency_connection_group_state_decoder,
            state_decoder_synapse.get_target_element().get_element_id(), id__)
        return state_decoder_synapse

    state_decoder_synapse_remove.__doc__ = \
        _StateDecoderSynapseMap.state_decoder_synapse_remove.__doc__ + \
        """

        提示:
            本方法扩展基类的同名方法，移除突触状态解码器对解码器模型和目标连接组的依赖关系。
        """

    def set_time_step(self, time_step: Union[float, Decimal]):
        """设置各个时间步共同的长度。

        参数:
            time_step: 各个时间步共同的长度。
        """
        if isinstance(time_step, Decimal):
            time_step_ = time_step
        else:
            time_step_ = float(time_step)
        if time_step_ <= 0:
            raise ValueError("time_step must be a positive real number.")
        self.__time_step = time_step_

    def get_time_step(self) -> Union[float, Decimal]:
        """获取各个时间步共同的长度。

        返回值:
            各个时间步共同的长度。
        """
        return self.__time_step

    def set_time_window(self, time_window: Union[float, Decimal]):
        """设置SNN模型运行时间窗口的长度。

        参数:
            time_window:
                SNN模型运行时间窗口的长度。`time_window` 为 ``None`` 表示SNN模型没有预
                先确定的运行时间窗口长度。

        提示:
            SNN模型的运行时间窗口是SNN模型的一次运行过程所经历的时间，该时间是SNN模型的时
            间，而不是真实世界的时间。
        """
        if time_window is None:
            time_window_ = None
        elif isinstance(time_window, Decimal):
            time_window_ = time_window
        else:
            time_window_ = float(time_window)
        if time_window_ < 0:
            raise ValueError(
                "time_window must be None, zero or a positive real number.")
        self.__time_window = time_window_

    def get_time_window(self) -> Union[float, Decimal]:
        """获取SNN模型运行时间窗口的长度。

        返回值:
            SNN模型运行时间窗口的长度。
        """
        return self.__time_window

    @staticmethod
    def __dependency_add(
            dependency_map: dict[str, set[str]], requirement: str,
            dependent: str):
        """将从被依赖的元素到依赖该元素的元素的映射项添加到依赖映射关系。

        参数:
            dependency_map: 依赖映射关系。
            requirement: 被依赖的元素。
            dependent: 依赖被依赖元素的元素。
        """
        dependent_set = dependency_map.get(requirement, None)
        dependent_set_added = False
        if dependent_set is None:
            dependent_set = set()
            dependency_map[requirement] = dependent_set
            dependent_set_added = True
        try:
            dependent_set.add(dependent)
        except:
            if dependent_set_added:
                dependency_map.pop(requirement, None)
            raise

    @staticmethod
    def __dependency_remove(
            dependency_map: dict[str, set[str]], requirement: str,
            dependent: str):
        """将从被依赖的元素到依赖该元素的元素的映射项从依赖映射关系移除。

        参数:
            dependency_map: 依赖映射关系。
            requirement: 被依赖的元素。
            dependent: 依赖被依赖元素的元素。
        """
        dependent_set = dependency_map.get(requirement, None)
        if dependent_set is None:
            return
        dependent_set.discard(dependent)
        if len(dependent_set) == 0:
            dependency_map.pop(requirement)
