from systems import ECGAN
import torch
from torch import Tensor
from torch.utils.data import Dataset

class GANDataset(Dataset):
  def __init__(self, gan: ECGAN, age: Tensor, sex: Tensor, dx: Tensor) -> None:
    with torch.no_grad():
      self.fake = gan(gan._get_noise(len(dx), (age, sex, dx)))
    self.age = age
    self.sex = sex
    self.dx = dx

  def __getitem__(self, i):
    return self.fake[i], self.age[i], self.sex[i], self.dx[i]
  
  def __len__(self):
    return len(self.fake)