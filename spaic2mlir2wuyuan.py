from mlir2wuyuan import mlir2wuyuan
from quantize import quantize_net
from spaic2mlir import spaic2mlir
from test import test
from tiny import TinyModel, ActorNetSpiking


def main():
  @test('构造 spaic ')
  def net():
    return TinyModel()

  @test('量化')
  def _():
    quantize_net(net)

  @test('转换 spaic -> mlir ')
  def net_op():
    return spaic2mlir(net)

  @test('转换 mlir -> wuyuan ')
  def net_wy():
    return mlir2wuyuan(net_op)


if __name__ == '__main__':
  main()
