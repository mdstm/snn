import numpy as np
import spaic


def get_value(a: spaic.BaseModule, var_name: str) -> np.ndarray:
  '''从后端提取数值，并转换为 numpy 类型'''
  return a._backend.get_varialble(var_name).numpy(force=True)


def get_neg_value(a: spaic.NeuronGroup, var: str) -> np.ndarray:
  '''从神经元组中提取数值，并转换为 numpy 类型'''
  return get_value(a, a.get_labeled_name(var))


def get_con_value(a: spaic.Connection, var: str) -> np.ndarray:
  '''从连接中提取数值，并转换为 numpy 类型'''
  return get_value(a, a.get_link_name(a.pre, a.post, var))


def update(d: dict, d1: dict):
  '''递归更新字典'''
  for k, v1 in d1.items():
    v = d.get(k)
    if isinstance(v, dict) and isinstance(v1, dict):
      update(v, v1)
    else:
      d[k] = v1
  return d
