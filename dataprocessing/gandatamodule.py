from .gandataset import DownstreamDataset
from.datamodule import PhysionetDataModule

import pytorch_lightning as pl
from systems import ECGAN
from torch.utils.data import DataLoader
from typing import Optional

class DownstreamDataModule(pl.LightningDataModule):
  def __init__(self, gan: ECGAN, pdm: PhysionetDataModule, 
               gen_mode: str, batch_size: int, num_workers: int) -> None:
    super().__init__()

    self.gan = gan
    self.pdm = pdm
    self.gen_mode = gen_mode
    self.batch_size = batch_size
    self.num_workers = num_workers
  
  def setup(self, stage: Optional[str] = None) -> None:
    if stage == "fit" or stage is None:
      self.train = DownstreamDataset(self.pdm.train, self.gan, self.gen_mode)
      self.val   = DownstreamDataset(self.pdm.val)
    if stage == "test" or stage is None:
      self.test  = DownstreamDataset(self.pdm.test)
  
  def _shared_dataloader(self, split: str):
    return DataLoader(getattr(self, split), batch_size=self.batch_size, num_workers=self.num_workers, 
                      pin_memory=True, shuffle=split == "train")
  
  def train_dataloader(self) -> DataLoader:
    return self._shared_dataloader("train")
  
  def val_dataloader(self) -> DataLoader:
    return self._shared_dataloader("val")

  def test_dataloader(self) -> DataLoader:
    return self._shared_dataloader("test")