from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import itertools

import numpy as np

from ..topology_model import TopologyModel


class Convolution2D(TopologyModel):
    __PARAMETERS = {
        "kernel_size": tuple, # 卷积核在竖直和水平两个方向上的大小。
        "padding": tuple,     # 输入特征图的填充量。若 padding 的长度为2，则其元素依次为竖直和水平两个方向上每边的填充量；若 padding 的长度为4，则其元素依次为上、下、左和右四个方向上的填充量。
        "stride": tuple,      # 竖直和水平两个方向上的步长。
        "dilation": tuple,    # 竖直和水平两个方向上的膨胀系数。
    }

    def __init__(self, presynaptic_shape: Sequence[int], postsynaptic_shape: Sequence[int], parameter: Mapping[str, Any]):
        """
        构造函数。

        Args:
            presynaptic_shape: 突触前节点的形状。
            postsynaptic_shape: 突触后节点的形状。
            parameter: 参数名称到参数值的映射关系。

        Raises:
            KeyError: 参数没有提供时抛出。
            TypeError: 参数类型不对时抛出。
            ValueError: 参数值不对时抛出。
        """
        super().__init__(presynaptic_shape, postsynaptic_shape)

        kernel_size = tuple(map(int, parameter["kernel_size"]))
        padding = tuple(map(int, parameter["padding"]))
        stride = tuple(map(int, parameter["stride"]))
        dilation = tuple(map(int, parameter["dilation"]))

        if len(padding) == 2:
            padding_0, padding_1 = padding
            padding = (padding_0, padding_0, padding_1, padding_1)

        pre_shape = self.get_presynaptic_shape()
        post_shape = self.get_postsynaptic_shape()
        if len(pre_shape) != 3:
            raise ValueError("presynaptic_shape's length should be 3.")
        if len(post_shape) != 3:
            raise ValueError("postsynaptic_shape's length should be 3.")

        if Convolution2D.find_post_shape(pre_shape[1:], kernel_size, padding,
                stride, dilation) != post_shape[1:]:
            raise ValueError("Convolution2D's shape is wrong.")

        self.set_parameter("kernel_size", kernel_size)
        self.set_parameter("padding", padding)
        self.set_parameter("stride", stride)
        self.set_parameter("dilation", dilation)

    @staticmethod
    def find_post_shape(
        pre_shape: tuple,
        kernel_size: tuple,
        padding: tuple,
        stride: tuple,
        dilation: tuple
    ) -> tuple[int, int]:
        """检察四种参数的形状和数值，并计算输出形状。"""

        if len(kernel_size) != 2:
            raise ValueError("kernel_size's length should be 2.")
        if len(padding) != 4:
            raise ValueError("padding's length should be 2 or 4.")
        if len(stride) != 2:
            raise ValueError("stride's length should be 2.")
        if len(dilation) != 2:
            raise ValueError("dilation's length should be 2.")

        if any(x < 1 for x in kernel_size):
            raise ValueError("kernel_size should not less than 1.")
        if any(x < 0 for x in padding):
            raise ValueError("padding should not less than 0.")
        if any(x <= y for x, y in zip(kernel_size, padding)):
            raise ValueError("padding should not larger than kernel size.")
        if any(x < 1 for x in stride):
            raise ValueError("stride should not less than 1.")
        if any(x < 1 for x in dilation):
            raise ValueError("dialation should not less than 1.")

        return (
            (pre_shape[0] + padding[0] + padding[1] \
             - ((kernel_size[0] - 1) * dilation[0] + 1)) // stride[0] + 1,
            (pre_shape[1] + padding[2] + padding[3] \
             - ((kernel_size[1] - 1) * dilation[1] + 1)) // stride[1] + 1,
        )

    @staticmethod
    def get_model_id() -> str:
        """
        获取模型的唯一标识符。
        """
        return "Convolution2D"

    def get_kernel_size(self) -> tuple[int, int]:
        """
        获取卷积核大小。
        """
        return self.get_original_parameter("kernel_size")

    def get_padding(self) -> tuple[int, int, int, int]:
        """
        获取输入特征图的填充量。
        """
        return self.get_original_parameter("padding")

    def get_stride(self) -> tuple[int, int]:
        """
        获取步长。
        """
        return self.get_original_parameter("stride")

    def get_dilation(self) -> tuple[int, int]:
        """
        获取膨胀系数。
        """
        return self.get_original_parameter("dilation")

    def get_shared_synapse_state_shape(self) -> tuple[int, ...]:
        """
        获取突触状态张量的形状。
        共享突触状态张量为4阶张量，其形状为 (Cout, Cin, Hk, Wk) ，其中 Cout 和 Cin 分别为输出通道数量和输入通道数量， Hk 和 Wk 分别为卷积核的高度和宽度。

        Returns:
            共享突触状态张量的形状
        """
        return self.get_instantiated_synapse_state_shape()

    def get_instantiated_synapse_state_shape(self) -> tuple[int, ...]:
        """
        获取突触状态张量的形状。

        Returns:
            突触状态张量的形状
        """
        return (self.get_postsynaptic_shape()[0],
            self.get_presynaptic_shape()[0]) + self.get_kernel_size()

    def get_connected_presynaptic_position(self, postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取与给定突触后神经元相连的突触前神经元。

        Args:
            postsynaptic_position: 矩阵，其中每行 postsynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            presynaptic_position: 每个元素 presynaptic_position[i] 是一个矩阵
            每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，该突触前神经元与 postsynaptic_position[i] 表示的突触后神经元相连

        """
        pre_positions = list()
        for post_position in postsynaptic_position:
            pre_positions.append(self.get_connected_presynaptic_position_single(post_position))
        return np.array(pre_positions)

    def get_connected_presynaptic_position_single(self, postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取与给定突触后神经元相连的突触前神经元。

        Args:
            postsynaptic_position: 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            presynaptic_position:
            每行 presynaptic_position[i] 是一个突触前神经元在其节点中的位置，该突触前神经元与 postsynaptic_position 表示的突触后神经元相连
        """
        # 确定左上角第一个元素(带padding) h,w
        # 序号转位置
        postsynaptic_position = self.check_and_convert_position(postsynaptic_position, self.get_postsynaptic_shape())

        pre_first_position = tuple(a * b for a, b in zip(postsynaptic_position[1:], self.__stride))
        pre_channel = self.get_presynaptic_shape()[0]
        pre_shape = self.get_presynaptic_shape()[1:]
        pre_positions = list()
        for i in range(self.__kernel_size[0]):
            for j in range(self.__kernel_size[1]):
                cur_pos = pre_first_position[0] + i * self.__dilation[0], pre_first_position[1] + j * self.__dilation[1]
                if cur_pos[0] < self.__padding[0] or cur_pos[0] >= self.__padding[0] + pre_shape[0] or cur_pos[1] < \
                        self.__padding[1] or cur_pos[1] >= self.__padding[1] + pre_shape[1]:
                    continue
                for c in range(pre_channel):
                    cur_pos = cur_pos[0] - self.__padding[0], cur_pos[1] - self.__padding[1]
                    pre_pos = (c,) + cur_pos
                    pre_positions.append(pre_pos)
        return np.array(pre_positions)

    def check_and_convert_position(self, position: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
        position_tuple = tuple(elem for elem in position)
        if len(position_tuple) > 1:
            return position_tuple
        index = position_tuple[0]
        prod = 1
        position_res = list()
        for elem in shape:
            prod *= elem
        for i in range(len(shape)):
            prod /= shape[i]
            position_res.append(index // prod)
            index %= prod
        return tuple(position_res)
    def get_connected_postsynaptic_position(self, presynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取与给定突触前神经元相连的突触后神经元。
        Args:
            presynaptic_position:矩阵，其中每行 presynaptic_position[i] 表示一个突触前神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            postsynaptic_position: 每个元素 postsynaptic_position[i] 是一个矩阵
            每行 postsynaptic_position[i][j] 是一个突触后神经元在其节点中的位置，该突触后神经元与 presynaptic_position[i] 表示的突触前神经元相连。

        """
        post_positions = list()
        for pre_position in presynaptic_position:
            post_positions.append(self.get_connected_postsynaptic_position_single(pre_position))
        return np.array(post_positions)

    def get_connected_postsynaptic_position_single(self, presynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取与给定突触前神经元相连的突触后神经元。
        Args:
            presynaptic_position: 表示一个突触前神经元。
            若列数大于1，则本方法将其视为一个神经元在其节点中的位置；
            若列数等于1，则本方法将其视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            postsynaptic_position:
            postsynaptic_position[i] 是一个突触后神经元在其节点中的位置，该突触后神经元与 presynaptic_position表示的突触前神经元相连。
        """
        # 序号转位置
        presynaptic_position = self.check_and_convert_position(presynaptic_position, self.get_presynaptic_shape())
        shape_len = len(presynaptic_position[1:])
        padding = tuple(self.__padding for i in range(0, len(self.__padding), 2))
        pre_position_with_padding = tuple(a + b for a, b in zip(presynaptic_position[1:], padding[:shape_len]))
        pre_shape = self.get_presynaptic_shape()[1:]
        post_channel = self.get_postsynaptic_shape()[0]
        post_shape = self.get_postsynaptic_shape()[1:]
        post_positions = list()
        for i in range(self.__kernel_size[0]):
            for j in range(self.__kernel_size[1]):
                # 计算当前位置与kernel（i，j）对应时的左上角位置
                first_position = pre_position_with_padding[0] - i * self.__dilation[0], pre_position_with_padding[
                    1] - j * self.__dilation[1]
                # 计算当前位置与kernel（i，j）对应时的右下角位置
                last_position = first_position[0] + (self.__kernel_size[0] - 1) * self.__dilation[0], first_position[
                    1] + (self.__kernel_size[1] - 1) * self.__dilation[1]
                # 校验范围
                if first_position[0] < 0 or first_position[1] < 0 or last_position[0] >= self.__padding[0] + \
                        self.__padding[1] + pre_shape[0] or last_position[1] >= self.__padding[2] + self.__padding[3] + \
                        pre_shape[1]:
                    continue
                if first_position[0] >= post_shape[0] or first_position[1] >= post_shape[1]:
                    continue
                # 校验 stride
                if first_position[0] % self.__stride[0] != 0 or first_position[1] % self.__stride[1] != 0:
                    continue
                for c in range(post_channel):
                    pos_position = (c,) + first_position
                    post_positions.append(pos_position)
        return np.array(post_positions)

    def get_input_connection_shared(self, postsynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触后神经元的输入连接。本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Args:
            postsynaptic_position:矩阵，其中每行 postsynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            presynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            presynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 postsynaptic_position[i] 表示的突触后神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        presynaptic_position = list()
        synapse_state_position = list()
        for pre_position in postsynaptic_position:
            res = self.get_input_connection_single(pre_position, True)
            presynaptic_position.append(res[0])
            synapse_state_position.append(res[1])
        return np.ndarray(presynaptic_position), np.ndarray(synapse_state_position)

    def get_input_connection_instantiated(self, postsynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触后神经元的输入连接。本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量

        Args:
            postsynaptic_position:矩阵，其中每行 postsynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            presynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            presynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 postsynaptic_position[i] 表示的突触后神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        presynaptic_position = list()
        synapse_state_position = list()
        for pre_position in postsynaptic_position:
            res = self.get_input_connection_single(pre_position, False)
            presynaptic_position.append(res[0])
            synapse_state_position.append(res[1])
        return np.ndarray(presynaptic_position), np.ndarray(synapse_state_position)

    def get_input_connection_single(self, postsynaptic_position: np.ndarray, is_shared: bool) -> tuple[
        np.ndarray, np.ndarray]:
        """
        获取给定突触后神经元的输入连接。

        Args:
            is_shared: 是否共享状态值
            postsynaptic_position: 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。

        Returns:
            presynaptic_position:
            每行 presynaptic_position[i] 是一个突触前神经元在其节点中的位置，该突触前神经元与 postsynaptic_position 表示的突触后神经元相连
            synapse_state_position:
            每行 synapse_state_position[i] 是一个突触状态值在突触状态张量中的位置，该突触状态值与 postsynaptic_position 表示的突触后神经元相连
        """
        # 确定左上角第一个元素(带padding) h,w
        # 序号转位置
        postsynaptic_position = self.check_and_convert_position(postsynaptic_position, self.get_postsynaptic_shape())

        pre_first_position = tuple(a * b for a, b in zip(postsynaptic_position[1:], self.__stride))
        pre_channel = self.get_presynaptic_shape()[0]
        pre_shape = self.get_presynaptic_shape()[1:]
        pre_positions = list()
        synapse_state_positions = list()
        for i in range(self.__kernel_size[0]):
            for j in range(self.__kernel_size[1]):
                cur_pos = pre_first_position[0] + i * self.__dilation[0], pre_first_position[1] + j * self.__dilation[1]
                if cur_pos[0] < self.__padding[0] or cur_pos[0] >= self.__padding[0] + pre_shape[0] or cur_pos[1] < \
                        self.__padding[1] or cur_pos[1] >= self.__padding[1] + pre_shape[1]:
                    continue
                for c in range(pre_channel):
                    cur_pos = cur_pos[0] - self.__padding[0], cur_pos[1] - self.__padding[1]
                    pre_pos = (c,) + cur_pos
                    if is_shared:
                        synapse_state_position = postsynaptic_position[0], i, j
                    else:
                        # 对于非共享情况(前神经元位置，后神经元位置）
                        synapse_state_position = pre_pos, postsynaptic_position
                    pre_positions.append(pre_pos)
                    synapse_state_positions.append(synapse_state_position)
        return np.array(pre_positions), np.array(synapse_state_positions)

    def get_output_connection_shared(self, presynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触前神经元的输出连接。本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Args:
            presynaptic_position:矩阵，其中每行 presynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            postsynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            postsynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 presynaptic_position[i] 表示的突触前神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        postsynaptic_position = list()
        synapse_state_position = list()
        for pre_position in presynaptic_position:
            res = self.get_input_connection_single(pre_position, True)
            postsynaptic_position.append(res[0])
            synapse_state_position.append(res[1])
        return np.ndarray(postsynaptic_position), np.ndarray(synapse_state_position)

    def get_output_connection_instantiated(self, presynaptic_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        获取给定突触前神经元的输出连接。本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Args:
            presynaptic_position:矩阵，其中每行 presynaptic_position[i] 表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            postsynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            postsynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 presynaptic_position[i] 表示的突触前神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        postsynaptic_position = list()
        synapse_state_position = list()
        for pre_position in presynaptic_position:
            res = self.get_input_connection_single(pre_position, False)
            postsynaptic_position.append(res[0])
            synapse_state_position.append(res[1])
        return np.ndarray(postsynaptic_position), np.ndarray(synapse_state_position)

    def get_output_connection_single(self, presynaptic_position: np.ndarray, is_shared: bool) -> tuple[
        np.ndarray, np.ndarray]:
        """
        获取给定突触前神经元的输出连接。本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Args:
            is_shared:
            presynaptic_position:表示一个突触后神经元。
            若该矩阵的列数大于1，则本方法将矩阵的每行视为一个神经元在其节点中的位置；
            若该矩阵的列数等于1，则本方法将矩阵的每行视为一个神经元在其节点中的序号。
            对于一维形状的节点，同一神经元在节点中的位置和序号相等。


        Returns:
            postsynaptic_position, synapse_state_position: 每个元素是一个矩阵。
            postsynaptic_position[i] 的每行 presynaptic_position[i][j] 是一个突触前神经元在其节点中的位置，
            该突触前神经元与 presynaptic_position[i] 表示的突触前神经元相连。
            对应的突触状态值在突触状态张量中的位置为 synapse_state_position[i][j] 。

        """
        # 序号转位置
        presynaptic_position = self.check_and_convert_position(presynaptic_position, self.get_presynaptic_shape())

        # presynaptic_position和__padding的维度问题
        shape_len = len(presynaptic_position[1:])
        padding = tuple(self.__padding[i] for i in range(0, len(self.__padding), 2))
        pre_position_with_padding = tuple(a + b for a, b in zip(presynaptic_position[1:], padding[:shape_len]))
        pre_shape = self.get_presynaptic_shape()[1:]
        post_channel = self.get_postsynaptic_shape()[0]
        post_shape = self.get_postsynaptic_shape()[1:]
        post_positions = list()
        synapse_state_positions = list()
        for i in range(self.__kernel_size[0]):
            for j in range(self.__kernel_size[1]):
                # 计算当前位置与kernel（i，j）对应时的左上角位置
                first_position = pre_position_with_padding[0] - i * self.__dilation[0], pre_position_with_padding[
                    1] - j * self.__dilation[1]
                # 计算当前位置与kernel（i，j）对应时的右下角位置
                last_position = first_position[0] + (self.__kernel_size[0] - 1) * self.__dilation[0], first_position[
                    1] + (self.__kernel_size[1] - 1) * self.__dilation[1]
                # 校验范围
                if first_position[0] < 0 or first_position[1] < 0 or last_position[0] >= self.__padding[0] + \
                        self.__padding[1] + pre_shape[0] or last_position[1] >= self.__padding[2] + self.__padding[3] + \
                        pre_shape[1]:
                    continue
                if first_position[0] >= post_shape[0] or first_position[1] >= post_shape[1]:
                    continue
                # 校验 stride
                if first_position[0] % self.__stride[0] != 0 or first_position[1] % self.__stride[1] != 0:
                    continue
                for c in range(post_channel):
                    pos_position = (c,) + first_position
                    if is_shared:
                        synapse_state_position = c, i, j
                    else:
                        # 对于非共享情况(前神经元位置，后神经元位置）
                        # 都是tuple
                        synapse_state_position = presynaptic_position + pos_position
                    post_positions.append(pos_position)
                    synapse_state_positions.append(synapse_state_position)
        return np.array(post_positions), np.array(synapse_state_positions)

    def get_shared_synapse_state_position(self, presynaptic_position: np.ndarray,
                                          postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取给定突触的状态值在连接组突触状态张量中的位置。

        Args:
            presynaptic_position: 有n行,每行表示一个突触前神经元的位置
            postsynaptic_position: 有n行,每行表示一个突触后神经元的位置

        Returns:
            synapse_state_position: 有n行,每行表示一个突触在状态张量中的位置

        """
        synapse_state_position = list()
        for i in range(len(presynaptic_position)):
            synapse_state_position.append(self.get_shared_synapse_state_position_single(
                presynaptic_position[i], postsynaptic_position[i]))
        return np.array(synapse_state_position)

    def get_shared_synapse_state_position_single(self, presynaptic_position: np.ndarray,
                                                 postsynaptic_position: np.ndarray) -> np.ndarray:
        shape_len = len(presynaptic_position[1:])
        padding = tuple(self.__padding[i] for i in range(0, len(self.__padding), 2))
        pre_first_position = tuple(a * b for a, b in zip(postsynaptic_position[1:], self.__stride))
        pre_position_with_padding = tuple(a + b for a, b in zip(presynaptic_position[1:], padding[:shape_len]))
        diff = tuple(a - b for a, b in zip(pre_position_with_padding, pre_first_position))
        for i in range(len(diff)):
            if diff[i] < 0 or diff[i] % self.__dilation[i] != 0:
                return np.fill(-1, len(diff))
            diff[i] %= self.__dilation[i]
        return np.concatenate((postsynaptic_position[0], diff))

    def get_shared_synapse_state_position_iterator(self) -> Iterator[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。

        本方法针对元素被不同突触共享的状态张量，而非连接组中所有突触实例的状态值组成的张量。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。

        """
        pre_ranges = [range(i) for i in self.__presynaptic_shape]
        pre_coordinates = list(itertools.product(*pre_ranges))
        post_ranges = [range(i) for i in self.__postsynaptic_shape]
        post_coordinates = list(itertools.product(*post_ranges))
        for pre_coord in pre_coordinates:
            for post_coord in post_coordinates:
                yield pre_coord, post_coord, tuple(
                    self.get_shared_synapse_state_position_single(pre_coord, post_coord))

    def get_instantiated_synapse_state_position(self, presynaptic_position: np.ndarray,
                                                postsynaptic_position: np.ndarray) -> np.ndarray:
        """
        获取给定突触的状态值在连接组突触状态张量中的位置。

        Args:
            presynaptic_position: 有n行,每行表示一个突触前神经元的位置
            postsynaptic_position: 有n行,每行表示一个突触后神经元的位置

        Returns:
            synapse_state_position: 有n行,每行表示一个突触在状态张量中的位置
        """
        return np.array(
            [np.concatenate((pre, post)) for pre, post in zip(presynaptic_position, postsynaptic_position)])

    def get_instantiated_synapse_state_position_iterator(self) -> Iterator[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        获取突触状态值位置迭代器。
        本方法针对连接组中所有突触实例的状态值组成的张量，而非元素被不同突触共享的状态张量。

        Returns:
            每次迭代一个突触，返回值依次包括：
            1.突触前神经元在其神经元组中的位置；
            2.突触后神经元在其神经元组中的位置；
            3.突触状态值在突触状态张量中的位置。

        """
        pre_ranges = [range(i) for i in self.__presynaptic_shape]
        pre_coordinates = list(itertools.product(*pre_ranges))
        post_ranges = [range(i) for i in self.__postsynaptic_shape]
        post_coordinates = list(itertools.product(*post_ranges))
        for pre_coord in pre_coordinates:
            for post_coord in post_coordinates:
                yield pre_coord, post_coord, pre_coord + post_coord

    def has_parameter(self, name: str) -> bool:
        """
        检查给定名称是否为有效的参数。

        Args:
            name (str): 要检查的参数名称。

        Returns:
            如果参数名称有效，则返回 True；否则返回 False。
        """
        return name in Convolution2D.__PARAMETERS
    
    def get_parameter_type(self, name: str) -> type:
        """
        获取指定参数的类型。

        Args:
            name (str): 参数名称。

        Raises:
            KeyError: 如果 name 不是有效的参数名称。

        Returns:
            参数的类型。
        """
        return Convolution2D.__PARAMETERS[name]

    def get_parameter_type_iterator(self) -> Iterator[tuple[str, type]]:
        """
        获取参数类型的迭代器。

        Returns:
            参数名称及其类型的迭代器。
        """
        return iter(Convolution2D.__PARAMETERS.items())

    def get_tuple(self):
        return (
            self.get_presynaptic_shape(),
            self.get_postsynaptic_shape(),
            self.get_kernel_size(),
            self.get_padding(),
            self.get_stride(),
            self.get_dilation(),
        )

    def __hash__(self):
        return hash(self.get_tuple())

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Convolution2D)
            and self.get_tuple() == other.get_tuple()
        )
