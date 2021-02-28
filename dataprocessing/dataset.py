import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate, resample
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Tuple
from utils import all_paths_exist, classes, rfreq, raw_data_path, num_classes

class PhysionetDataset(Dataset):
  def __init__(self, df: pd.DataFrame, rdim: Tuple, hint_N: int = 2 ** 14) -> None:
    self.orig = torch.empty(hint_N, *rdim)
    self.pid = torch.empty(hint_N, dtype=torch.long)
    self.age = torch.empty(hint_N, 1)
    self.sex = torch.empty(hint_N, 1)
    self.dx = torch.empty(hint_N, num_classes)

    r = 0
    for i, entry in tqdm(enumerate(df.iloc)):
      age = entry["Age"]
      sex = entry["Gender_Male"]
      if not np.isfinite(age) or not np.isfinite(sex) or (sex != 0 and sex != 1): continue

      recording = PhysionetDataset._read_recording(entry["Patient"], rdim)
      if recording is None: continue
      
      dx = torch.tensor(df.iloc[i][classes].astype(np.int).to_numpy()).unsqueeze(0)
      N = len(recording)
      while r + N > self.orig.shape[0]: self._grow()

      self.orig[r:r+N] = recording
      self.pid[r:r+N] = i
      self.age[r:r+N] = age
      self.sex[r:r+N] = sex
      self.dx[r:r+N] = dx

      r += N
    self._trim(r)
    self.age = (self.age - self.age.min()) / (self.age.max() - self.age.min())

  def __getitem__(self, i):
    #pid keeps track of which crops came from which patients in case we need it
    return self.orig[i], self.pid[i], self.age[i], self.sex[i], self.dx[i]
  
  def __len__(self):
    return len(self.orig)
  
  def _grow(self):
    self.orig = torch.cat((self.orig, torch.empty_like(self.orig)), dim=0)
    self.pid = torch.cat((self.pid, torch.empty_like(self.pid)), dim=0)
    self.age = torch.cat((self.age, torch.empty_like(self.age)), dim=0)
    self.sex = torch.cat((self.sex, torch.empty_like(self.sex)), dim=0)
    self.dx = torch.cat((self.dx, torch.empty_like(self.dx)), dim=0)

  def _trim(self, r: int):
    self.orig = self.orig[:r].contiguous()
    self.pid = self.pid[:r].contiguous()
    self.dx = self.dx[:r].contiguous()

  @staticmethod
  def _read_recording(id: str, rdim: Tuple) -> Optional[Tensor]:
    file_name = f"{raw_data_path}/{id}"
    rC, rL = rdim

    recording = PhysionetDataset._process_recording(file_name, rL)

    C, L = recording.shape
    if C != rC or L < rL or not torch.all(recording.isfinite()): return None
    recording = recording[:, :rL * (L // rL)]
    
    recording = recording.view(C, -1, rL).transpose(0, 1) #exhaustive crop
    if not torch.all(torch.isfinite(recording)): recording = None
    return recording

  @staticmethod
  def _process_recording(file_name: str, rL: int):
    recording = loadmat(f"{file_name}.mat")['val'].astype(float)

    # Standardize sampling rate
    sampling_rate = PhysionetDataset._get_sampling_rate(file_name)

    if sampling_rate > rfreq:
      recording = np.copy(decimate(recording, int(sampling_rate / rfreq)))
    elif sampling_rate < rfreq:
      recording = np.copy(resample(recording, int(recording.shape[-1] * (rfreq / sampling_rate)), axis=1))
    
    return torch.from_numpy(PhysionetDataset._normalize(recording))
  
  @staticmethod
  def _normalize(x: np.ndarray):
    return x / (np.max(x) - np.min(x) + 1e-8)
  
  @staticmethod
  def _get_sampling_rate(file_name: str):
    with open(f"{file_name}.hea", 'r') as f:
      return int(f.readline().split(None, 3)[2])