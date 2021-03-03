from models import Generator, Discriminator
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce, mse_loss as mse
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

class ECGAN(pl.LightningModule):
  def __init__(self, config: Dict) -> None:
    super().__init__()
    self.save_hyperparameters(config)

    #Packs noise, age, sex, and class info into one vector
    self.gen = Generator(self.hparams.C, self.hparams.L, 
      self.hparams.z_dim + 1 + 1 + self.hparams.num_classes)
    
    #Classifiers for Source, Age, Sex, and Class, in that order
    self.disc = Discriminator(self.hparams.C, outs = [1, 1, 1, self.hparams.num_classes]) 
  
  def forward(self, age: Tensor, sex: Tensor, dx: Tensor, downstream: bool = False) -> Tensor:
    if downstream:
      output = torch.empty(0)
      for i in tqdm(range(0, len(dx), self.hparams.batch_size)):
        end = min(i + self.hparams.batch_size, len(dx))
        output = torch.cat((output, 
          self.gen(self._get_input([age[i:end], sex[i:end], dx[i:end]])).cpu()), dim=0)
      return output
    return self.gen(self._get_input([age, sex, dx])) #useful for generating datasets all at once

  def training_step(self, batch: Tuple, batch_idx: int, optimizer_idx: int):
    if optimizer_idx == 0: return self._discriminator_step(batch) 
    elif optimizer_idx == 1: return self._generator_step(batch)
    else: raise ValueError("Unhandled optimizer")
  
  """
    @fn   _generator_step: Calculates the loss in the generator step.
          Generates as many fake examples as real examples, conditioned 
          on whatever labels the real examples have
    @param batch: A tuple of tensors output by the dataloader
  """
  def _generator_step(self, batch):
    real, pid, *lbls = batch
    fake = self.gen(self._get_input(lbls))

    return self._calculate_loss("gen", self.disc(fake), lbls) #pass fake in as if it were real

  """
    @fn    _discriminator_step: Calculates the loss in the discriminator step.
    @param batch: A tuple of tensors output by the dataloader
  """
  def _discriminator_step(self, batch):
    real, pid, *lbls = batch

    with torch.no_grad():
      fake = self.gen(self._get_input(lbls))
    
    return self._calculate_loss("disc", self.disc(real), lbls, self.disc(fake))
  
  """
    @fn    _get_input: Get input vector to for GAN.
    @param N: How many samples are going to be generated
    @param lbls: a tuple of labels containing age, sex, and diagnosis in that order.
                 If None, random noise is passed in as labels.
  """
  def _get_input(self, lbls: List[Tensor]):
    N = self.check_lbls(lbls)
    return torch.cat((torch.rand(N, self.hparams.z_dim, device=self.device), *lbls), dim=1)

  """
    @fn    _calculate_loss: Calculates loss inspired from AC-GAN paper
    @param stage: a string describing the current step of training (disc or gen)
    @param r: a tuple of discriminator outputs from real examples
    @param lbls: a tuple of labels containing age, sex, and diagnosis in that order
    @param f: an optional tuple of discriminator outputs from fake examples
  """
  def _calculate_loss(self, stage: str, 
                      r: Tuple[Tensor], lbls: List[Tensor], 
                      f: Optional[Tuple[Tensor]] = None):
    r_pred, r_age, r_sex, r_dx = r #unpack discriminator outputs for reals
    age, sex, dx = lbls

    #bce loss for every output except age, which uses mse because it is a continuous variable
    Ls = bce(r_pred, torch.ones_like(r_pred))
    Lc = mse(r_age, age) + bce(r_sex, sex) + bce(r_dx , dx)
    
    if f: #can optionally pass in fakes
      f_pred, f_age, f_sex, f_dx = f #unpack discriminator outputs for fakes
      Ls = Ls + bce(f_pred, torch.zeros_like(f_pred))
      Lc = Lc + mse(f_age, age) + bce(f_sex, sex) + bce(f_dx, dx)

    loss = Ls + Lc #if called correctly, this is the same for both disc and gen
    self.log(f"{stage}_loss", loss) #log loss
    return loss
  
  def check_lbls(self, lbls: List[Tensor]):
    N = lbls[0].shape[0]
    as_shape = torch.Size((N, 1))
    dx_shape = torch.Size((N, self.hparams.num_classes))
    if len(lbls) != 3 or lbls[0].shape != lbls[1].shape != as_shape or \
       lbls[2].shape != dx_shape:
      raise ValueError("Incorrect labels passed in.\n"
                       f"age: saw {lbls[0].shape}, expected {as_shape}\n"
                       f"sex: saw {lbls[1].shape}, expected {as_shape}\n"
                       f" dx: saw {lbls[2].shape}, expected {dx_shape}")
    return N
  
  def configure_optimizers(self):
    return [
      Adam(self.disc.parameters(), lr=self.hparams.lr), 
      Adam(self.gen.parameters(), lr=self.hparams.lr)
    ]
  
  def enable_downstream(self):
    del self.disc
