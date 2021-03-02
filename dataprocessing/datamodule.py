from .dataset import PhysionetDataset

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from utils import equiv_class_groups

class PhysionetDataModule(pl.LightningDataModule):
  def __init__(self, path: str, frac: float, frac_mode: str, folds: Dict, dims: Tuple, 
               batch_size: int,  num_workers: int) -> None:
    super().__init__()

    self.frac = frac
    self.frac_mode = frac_mode
    self.splits = PhysionetDataModule._get_splits(path)
    self.folds = folds
    self.dims = dims
    self.batch_size = batch_size
    self.num_workers = num_workers
  
  def setup(self, stage: Optional[str] = None) -> None:
    df = lambda split: self.splits[self.splits["fold"].isin(self.folds[split])]
    if stage == "fit" or stage is None:
      self.train = PhysionetDataset(df("train"), self.dims, self.frac, self.frac_mode)
      self.val   = PhysionetDataset(df("val"), self.dims)
    if stage == "test" or stage is None:
      self.test  = PhysionetDataset(df("test"), self.dims)
  
  def _shared_dataloader(self, split: str):
    return DataLoader(getattr(self, split), batch_size=self.batch_size, num_workers=self.num_workers, 
                      pin_memory=True, shuffle=split == "train")
  
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


