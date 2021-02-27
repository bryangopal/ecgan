from torch import nn, Tensor
from typing import List

class Discriminator(nn.Module):
  """
    @fn    __init__
    @param C_in: number of input channels
    @param outs: list of output dimensions of every intended classifier
  """
  def __init__(self, C_in: int = 12, outs: List[int] = [1, 24]) -> None:
    super().__init__()

    self.disc = nn.Sequential(
      Discriminator.make_disc_block(C_in, 64),
      Discriminator.make_disc_block(64, 128),
      Discriminator.make_disc_block(128, 256),
      Discriminator.make_disc_block(256, 512),
      Discriminator.make_disc_block(512, 1024),
      Discriminator.make_disc_block(1024, 2048, final=True)
    )
    self.avg = nn.AdaptiveAvgPool1d(1)
    self.flatten = nn.Flatten()
    self.fcs = nn.ModuleList([nn.Linear(2048, out) for out in outs])

  @staticmethod
  def make_disc_block(C_in: int, C_out: int, kernel_size: int = 3, stride: int = 2, final: bool = False):
    return nn.Sequential(
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, bias=False),
      nn.BatchNorm1d(C_out),
      nn.LeakyReLU(0.2, inplace=True)
    ) if not final else nn.Sequential(
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride)
    )

  def forward(self, x: Tensor) -> Tensor:
    x = self.disc(x)
    x = self.avg(x)
    x = self.flatten(x)
    return tuple(fc(x) for fc in self.fcs)