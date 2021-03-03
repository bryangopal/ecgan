from typing import Optional
from .dataset import PhysionetDataset
from systems import ECGAN
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

class DownstreamDataset(Dataset):
  def __init__(self, pds: PhysionetDataset, gan: Optional[ECGAN] = None, gen_mode: Optional[str] = None) -> None:
    super().__init__()
    self.orig = pds.orig
    self.age = pds.age
    self.sex = pds.sex
    self.dx = pds.dx
    if gan and gen_mode: self._gen_data(gan, pds.n_removed, gen_mode)

  def _gen_data(self, gan: ECGAN, N: int, gen_mode: str):
    age, sex, dx = None, None, None
    if gen_mode == "replace":
      self.orig = gan(self.age, self.sex, self.dx, downstream=True)
      return
    elif gen_mode == "parity":
      age = torch.rand(N, 1)
      sex = torch.randint(high=2, size=(N, 1), dtype=torch.float)
      dx = F.one_hot(torch.randint(high=self.dx.shape[1], size=(N,)))
    else: raise ValueError(f"Unsupported generation mode: {gen_mode}")

    self.orig = torch.cat((self.orig, gan(age, sex, dx, downstream=True)), dim=0)
    self.age  = torch.cat((self.age, age), dim=0)
    self.dx   = torch.cat((self.dx, dx), dim=0)

  def __getitem__(self, i):
    return self.fake[i], self.age[i], self.sex[i], self.dx[i]
  
  def __len__(self):
    return len(self.fake)