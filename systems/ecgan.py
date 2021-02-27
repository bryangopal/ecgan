from models import Generator, Discriminator
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce, mse_loss as mse
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple

class ECGAN(pl.LightningModule):
  def __init__(self, config: Dict) -> None:
    super().__init__()
    self.save_hyperparameters(config)

    self.gen = Generator(self.hparams.C, self.hparams.L, 
      self.hparams.z_dim + 1 + 1 + self.hparams.num_classes)
    self.disc = Discriminator(self.hparams.C, outs = [1, 1, 1, self.hparams.num_classes])

  
  def forward(self, z: Tensor) -> Tensor:
    return self.gen(z)
  
  def training_step(self, batch: Tuple, batch_idx: int, optimizer_idx: int):
    if optimizer_idx == 0: return self._discriminator_step(batch)
    elif optimizer_idx == 1: return self._generator_step(batch)
    else: raise ValueError("Unhandled optimizer")
  
  def _generator_step(self, batch):
    real, pid, *lbls = batch
    fake = self.gen(self._get_noise(len(real), lbls))

    return self._calculate_loss("gen", self.disc(fake), lbls) #pass fake in as if it were real

  def _discriminator_step(self, batch):
    real, pid, *lbls = batch

    with torch.no_grad():
      fake = self.gen(self._get_noise(len(real), lbls))
    
    return self._calculate_loss("disc", self.disc(real), lbls, self.disc(fake))
  
  def _get_noise(self, N, lbls: List[Tensor]):
    return torch.cat((torch.rand(N, self.hparams.z_dim, device=self.device), *lbls), dim=1)

  def _calculate_loss(self, stage: str, 
                      r: Tuple[Tensor], lbls: List[Tensor], 
                      f: Optional[Tuple[Tensor]] = None):
    r_pred, r_age, r_sex, r_dx = r
    age, sex, dx = lbls

    Ls = bce(r_pred, torch.ones_like(r_pred))
    Lc = mse(r_age, age) + bce(r_sex, sex) + bce(r_dx , dx)
    
    if f:
      f_pred, f_age, f_sex, f_dx = f
      Ls = Ls + bce(f_pred, torch.zeros_like(f_pred))
      Lc = Lc + mse(f_age, age) + bce(f_sex, sex) + bce(f_dx, dx)

    loss = Ls + Lc
    self.log(f"{stage}_loss", loss)
    return loss

  def configure_optimizers(self):
    return [
      Adam(self.disc.parameters(), lr=self.hparams.lr), 
      Adam(self.gen.parameters(), lr=self.hparams.lr)
    ]
  
  def enable_downstream(self):
    del self.disc