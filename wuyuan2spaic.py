import spaic

import wuyuan.snn_model as wym

from quantize import set_value
from wuyuan_to_spaic_info import get_infos


def wuyuan2spaic(
  net_wy: wym.Network,
  name: str,
  b: spaic.Torch_Backend,
) -> spaic.Network:
  '''把 wuyuan 表示的 SNN 转换为 spaic'''

  infos = get_infos(net_wy)

  net = spaic.Network(name)

  # 存储已构造的编码器、神经元组和连接，用于构造连接、解码器和监视器
  elements: dict[str] = {}

  # 需要在后端设置的变量
  back_neg: list[tuple[spaic.NeuronGroup, list[tuple]]] = []
  back_con: list[tuple[spaic.Connection, list[tuple]]] = []

  for name, info in infos.items():
    match info['type']:
      case 'Encoder':
        a = spaic.Encoder(**info['param'])
        setattr(net, name, a)
        elements[name] = a
      case 'NeuronGroup':
        a = spaic.NeuronGroup(**info['param'])
        setattr(net, name, a)
        elements[name] = a
        back_neg.append((a, info['backend']))
      case 'Connection':
        a = spaic.Connection(pre=elements[info['pre']],
          post=elements[info['post']], **info['param'])
        setattr(net, name, a)
        elements[name] = a
        back_con.append((a, info['backend']))
      case 'Decoder':
        setattr(net, name, spaic.Decoder(dec_target=elements[info['target']],
          **info['param']))
      case 'StateMonitor':
        setattr(net, name, spaic.StateMonitor(target=elements[info['target']],
          **info['param']))

  net.build(b)
  for a, vars in back_neg:
    for var, value in vars:
      set_value(b, a.get_labeled_name(var), value)
  for a, vars in back_con:
    for var, value in vars:
      set_value(b, a.get_link_name(a.pre, a.post, var), value)

  return net


# 测试

def main():
  from xdsl.ir import MLContext

  from mlir2wuyuan import mlir2wuyuan
  import snn
  from spaic2mlir import spaic2mlir
  from test import test
  from xdslm import Parser, Printer

  ctx = MLContext()
  ctx.load_dialect(snn.SNN)

  @test('读取 mlir ')
  def text():
    with open('spaic2mlir.mlir', 'r') as f:
      return f.read()

  @test('解析 mlir ')
  def net_op():
    return Parser(ctx, text).parse_op()

  @test('转换 mlir -> wuyuan ')
  def net_wy():
    return mlir2wuyuan(net_op)
  
  @test('转换 wuyuan -> spaic ')
  def net():
    b = spaic.Torch_Backend()
    b.dt = 1
    return wuyuan2spaic(net_wy, net_op.sym, b)

  @test('转换 spaic -> mlir ')
  def net_op_():
    return spaic2mlir(net)

  @test('打印 mlir ')
  def _():
    with open('wuyuan2spaic.mlir', 'w') as f:
      p = Printer(f)
      p.print_op(net_op_)
      p.print_string('\n')

  return net_op, net_wy, net, net_op_


if __name__ == '__main__':
  main()
