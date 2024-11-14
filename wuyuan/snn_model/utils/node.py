from collections.abc import Sequence

import numpy as np


def check_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """检查给定的节点形状是否合理。

    参数:
        shape: 节点形状。

    返回值:
        经过检查的合理的节点形状。

    提示:
        若给定的节点形状不合理，则本方法抛出ValueError。
    """
    shape_ = np.asarray(shape).reshape(-1)
    if not (
            np.issubsctype(shape_, np.integer) and shape_.size > 0 and
            np.all(shape_ > 0)):
        raise ValueError(
            "shape must be a non-empty sequence of positive integers.")
    return tuple(int(size) for size in shape_)


def get_neuron_position(shape: Sequence[int], neuron: np.ndarray) \
        -> np.ndarray:
    """获取神经元在节点中的位置。

    参数:
        shape: 节点的形状。
        neuron: 神经元在节点内部的位置或序号。

    返回值:
        神经元在节点中的位置。

    提示:
        `neuron` 是一个整数矩阵或整数向量。若 `neuron` 是矩阵，则每行给出一个神经元在节点
        中的位置；若 `neuron` 是向量，则每个元素给出一个神经元在节点内部的序号。

        本方法返回一个整数矩阵，每行给出一个神经元在节点中的位置。本方法返回的各个神经元位置与
        `neuron` 给出的各个神经元依次对应，若神经元不存在，则对应神经元位置的各个元素均为-1。
    """
    shape_ = check_shape(shape)
    neuron_ = np.asarray(neuron)
    if not np.issubsctype(neuron_, np.integer):
        raise TypeError("neuron must be an array of integers.")
    if not (neuron_.ndim == 1 or neuron_.ndim == 2):
        raise ValueError(
            "neuron must be either a one-dimensional array or a "
            "two-dimensional array.")
    position = np.full((neuron_.shape[0], len(shape_)), -1, dtype=np.int64)
    if neuron_.ndim == 1:
        # 查找序号在节点序号范围内的神经元。
        num_neuron_node = np.prod(shape_)
        existing = np.logical_and(neuron_ >= 0, neuron_ < num_neuron_node)
        position[existing, :] = \
            np.array(np.unravel_index(neuron_[existing], shape_)).T
        return position
    if neuron_.shape[1] != len(shape_):
        raise ValueError(
            "If neuron is a two-dimensional array, the number of columns must "
            "be equal to the number of dimensions of the node.")
    # 查找位置在节点位置范围内的神经元。
    existing = np.logical_and(neuron_ >= 0, neuron_ < shape_).all(axis=1)
    position[existing, :] = neuron_[existing, :]
    return position


def get_neuron_index(shape: Sequence[int], neuron: np.ndarray) -> np.ndarray:
    """获取神经元在节点内部的序号。

    参数:
        shape: 节点的形状。
        neuron: 神经元在节点内部的位置或序号。

    返回值:
        神经元在节点内部的序号组成的向量。

    提示:
        `neuron` 是一个整数矩阵或整数向量。若 `neuron` 是矩阵，则每行给出一个神经元在节点
        中的位置；若 `neuron` 是向量，则每个元素给出一个神经元在节点内部的序号。

        本方法返回的各个神经元序号与 `neuron` 给出的各个神经元依次对应，若神经元不存在，则对
        应的神经元序号为-1。
    """
    shape_ = check_shape(shape)
    neuron_ = np.asarray(neuron)
    if not np.issubsctype(neuron_, np.integer):
        raise TypeError("neuron must be an array of integers.")
    if not (neuron_.ndim == 1 or neuron_.ndim == 2):
        raise ValueError(
            "neuron must be either a one-dimensional array or a "
            "two-dimensional array.")
    index = np.full((neuron_.shape[0],), -1, dtype=np.int64)
    if neuron_.ndim == 1:
        # 查找序号在节点序号范围内的神经元。
        num_neuron_node = np.prod(shape_)
        existing = np.logical_and(neuron_ >= 0, neuron_ < num_neuron_node)
        index[existing] = neuron_[existing]
        return index
    if neuron_.shape[1] != len(shape_):
        raise ValueError(
            "If neuron is a two-dimensional array, the number of columns must "
            "be equal to the number of dimensions of the node.")
    # 查找位置在节点位置范围内的神经元。
    existing = np.logical_and(neuron_ >= 0, neuron_ < shape_).all(axis=1)
    index[existing] = np.ravel_multi_index(neuron_[existing, :].T, shape_)
    return index
