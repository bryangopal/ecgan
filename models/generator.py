from torch import nn, Tensor

class Generator(nn.Module):
  """
    @fn    __init__
    @param C_out: number of output channels
    @param L: length of final generated sample
    @param z_dim: dimensionality of noise vector
  """
  def __init__(self, C_out: int = 12, L: int = 2500, z_dim: int = 256) -> None:
    super().__init__()

    self.L = L
    self.l1 = nn.Linear(z_dim, z_dim)
    self.gen = nn.Sequential(
      Generator.make_gen_block(z_dim, 1024),
      Generator.make_gen_block(1024, 512),
      Generator.make_gen_block(512, 256),
      Generator.make_gen_block(256, 128),
      Generator.make_gen_block(128, 64),
      Generator.make_gen_block(64, C_out, final=True)
    )

  @staticmethod
  def make_gen_block(C_in: int, C_out: int, kernel_size: int = 8, stride: int = 3, final: bool = False):
    return nn.Sequential(
      nn.ConvTranspose1d(C_in, C_out, kernel_size, stride=stride, bias=False),
      nn.BatchNorm1d(C_out),
      nn.ReLU(inplace=True)
    ) if not final else nn.Sequential(
      nn.ConvTranspose1d(C_in, C_out, kernel_size, stride=stride),
      nn.Tanh()
    )

  def forward(self, z: Tensor) -> Tensor:
    return self.gen(self.l1(z).unsqueeze_(-1))[...,:self.L]