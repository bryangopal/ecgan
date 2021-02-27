from .dataset import PhysionetDataset

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Callable, Dict, Tuple, Optional
from utils import saved_data_path, equiv_class_groups

class PhysionetDataModule(pl.LightningDataModule):
  def __init__(self, path: str, folds: Dict, dims: Tuple, batch_size: int,  num_workers: int) -> None:
    super().__init__()

    self.splits = PhysionetDataModule._get_splits(path)
    self.folds = folds
    self.dims = dims
    self.batch_size = batch_size
    self.num_workers = num_workers
  
  def setup(self, stage: Optional[str] = None) -> None:
    if stage == "fit" or stage is None:
      self._load_dataset("train")
      self._load_dataset("val")
    if stage == "test" or stage is None:
      self._load_dataset("test")
  
  def _load_dataset(self, stage: str):
    df = self.splits[self.splits["fold"].isin(self.folds[stage])]
    print(f"Loading {stage} dataset...", end='', flush=True)
    setattr(self, stage, PhysionetDataset(df, f"{saved_data_path}/{stage}", self.dims))
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
  
  def get_labels(self):
    if not self.has_setup_fit: self.setup("fit")
    if not self.has_setup_test: self.setup("test")
    return [{
      "train": self.train.age,
      "val": self.val.age,
      "test": self.test.age
    }, {
      "train": self.train.sex,
      "val": self.val.sex,
      "test": self.test.sex
    }, {
      "train": self.train.dx,
      "val": self.val.dx,
      "test": self.test.dx
    }]
  
  @staticmethod
  def _get_splits(path: str):
    splits = pd.read_csv(f"{path}/splits.csv", index_col=0)
    for dx, dup_dxs in equiv_class_groups.items():
      for dup in dup_dxs:
        splits[dx] |= splits[dup]
      splits = splits.drop(columns=dup_dxs)
    return splits


