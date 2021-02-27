import models
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from torch.optim import Adam
from typing import Dict, List, Tuple
from utils import multiclass_auroc

class Classifier(pl.LightningModule):
  def __init__(self, config: Dict):
    super().__init__()
    self.save_hyperparameters(config)
    self.model = getattr(models, self.hparams.encoder)(self.hparams.num_channels)
    self.classifier = nn.Linear(2048, self.hparams.num_classes)
  
  def _shared_step(self, batch, stage: str):
    x, *_, y = batch

    y_hat = self.classifier(self.model(x))
    loss = bce(y_hat, y)

    self.log(f"{stage}_loss", loss)
    return loss if stage == "train" else (y_hat, y)
  
  def _shared_epoch_end(self, outputs: List[Tuple], stage: str):
    outputs = tuple(zip(*outputs))
    y_hat = torch.cat(outputs[0], dim=0)
    y = torch.cat(outputs[1], dim=0)

    self.log(f"{stage}_auroc", multiclass_auroc(y_hat, y))

  def training_step(self, batch, batch_idx):
    return self._shared_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self._shared_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self._shared_step(batch, "test")
  
  def validation_epoch_end(self, outputs: List[Tuple]):
    self._shared_epoch_end(outputs, "val")
  
  def test_epoch_end(self, outputs: List[Tuple]):
    self._shared_epoch_end(outputs, "test")
  
  def configure_optimizers(self):
    return Adam(self.parameters(), self.hparams.lr)