import models
from systems import ECGAN
import pytorch_lightning as pl
from pytorch_lightning.metrics import AUROC
import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits as bce, one_hot, softmax
from torch.optim import Adam
from typing import Dict, List, Tuple

class Classifier(pl.LightningModule):
  def __init__(self, config: Dict):
    super().__init__()
    self.save_hyperparameters(config)

    self.gan = ECGAN.load_from_checkpoint(self.hparams.gan_path)
    self.gan.enable_downstream()

    self.model = getattr(models, self.hparams.encoder)(self.hparams.num_channels)
    self.classifier = nn.Linear(2048, self.hparams.num_classes)
    
    if self.hparams.gen_mode is None: self.batch_fn = lambda batch: (batch[0], batch[-1]) 
    elif self.hparams.gen_mode == "replace": self.batch_fn = self.replace_batch
    elif self.hparams.gen_mode == "augment": self.batch_fn = self.augment_batch
    else: raise ValueError(f"Invalid generation mode: {self.hparams.gen_mode}")

    self.aurocs = {
      "val": AUROC(self.hparams.num_classes, compute_on_step=False, average="macro"),
      "test": AUROC(self.hparams.num_classes, compute_on_step=False, average="macro")
    }
  
  def _shared_step(self, batch, stage: str):
    x, y = self.batch_fn(batch)

    y_hat = self.classifier(self.model(x))
    loss = torch.mean(torch.stack([bce(y_hat[:, c], y[:, c]) for c in range(self.hparams.num_classes)]))

    self.log(f"{stage}_loss", loss)
    if stage == "train": return loss
    else: self.aurocs[stage](softmax(y_hat.detach(), dim=-1), y.long())
  
  def _shared_epoch_end(self, stage: str):
    self.log(f"{stage}_auroc", self.aurocs[stage].compute())
    self.aurocs[stage].reset()
  
  def replace_batch(self, batch):
    *_, age, sex, dx = batch
    return self.gan(age, sex, dx), dx
  
  def augment_batch(self, batch: Tuple[Tensor, ...]):
    orig, pid, age, sex, dx = batch
    N = int(self.hparams.gen_frac * len(dx))

    gan_dx = one_hot(
      torch.randint(high=self.hparams.num_classes, size=(N,), device=self.device)
    ).to(dx.dtype)

    with torch.no_grad():
      fake = self.gan(
        torch.rand(N, 1, device=self.device, dtype=age.dtype), 
        torch.randint(high=2, size=(N, 1), device=self.device, dtype=sex.dtype), 
        gan_dx
      )
    return torch.cat((orig, fake), dim=0), torch.cat((dx, gan_dx), dim=0)

  def training_step(self, batch, batch_idx):
    return self._shared_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self._shared_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self._shared_step(batch, "test")
  
  def validation_epoch_end(self, outputs):
    self._shared_epoch_end("val")
  
  def test_epoch_end(self, outputs):
    self._shared_epoch_end("test")
  
  def configure_optimizers(self):
    return Adam(list(self.model.parameters()) + list(self.classifier.parameters()), self.hparams.lr)