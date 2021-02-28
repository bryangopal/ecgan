from .gandataset import GANDataset

import pandas as pd
import pytorch_lightning as pl
from systems import ECGAN
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable, Dict, Tuple, Optional
from utils import equiv_class_groups

class GANDataModule(pl.LightningDataModule):
  def __init__(self, gan: ECGAN, ages: Dict[str, Tensor], 
               sexes: Dict[str, Tensor], dxs: Dict[str, Tensor], 
               batch_size: int, num_workers: int) -> None:
    super().__init__()

    self.gan = gan
    self.ages = ages
    self.sexes = sexes
    self.dxs = dxs
    self.batch_size = batch_size
    self.num_workers = num_workers
  
  def setup(self, stage: Optional[str] = None) -> None:
    if stage == "fit" or stage is None:
      self._load_dataset("train")
      self._load_dataset("val")
    if stage == "test" or stage is None:
      self._load_dataset("test")
  
  def _load_dataset(self, stage: str):
    print(f"Loading {stage} dataset...", end='', flush=True)
    setattr(self, stage, GANDataset(self.gan, self.ages[stage], self.sexes[stage], self.dxs[stage]))
    print(f"done: {getattr(self, stage).orig.shape}")
  
  def _shared_dataloader(self, stage: str):
    return DataLoader(getattr(self, stage), batch_size=self.batch_size, num_workers=self.num_workers, 
                      pin_memory=True, shuffle=stage == "train")
  
  def train_dataloader(self) -> DataLoader:
    return self._shared_dataloader("train")
  
  def val_dataloader(self) -> DataLoader:
    return self._shared_dataloader("val")

  def test_dataloader(self) -> DataLoader:
    return self._shared_dataloader("test")
  
  @staticmethod
  def _get_splits(path: str):
    splits = pd.read_csv(f"{path}/splits.csv", index_col=0)
    for dx, dup_dxs in equiv_class_groups.items():
      for dup in dup_dxs:
        splits[dx] |= splits[dup]
      splits = splits.drop(columns=dup_dxs)
    return splits


