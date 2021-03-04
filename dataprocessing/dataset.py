import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate, resample
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Tuple
from utils import classes, rfreq, raw_data_path, num_classes

class PhysionetDataset(Dataset):
  def __init__(self, df: pd.DataFrame, rdim: Tuple, replace: bool, frac: Optional[float] = None, 
               frac_mode: Optional[str] = None, hint_N: int = 2 ** 14) -> None:
    super().__init__()
    self.orig = torch.empty(hint_N, *rdim) if not replace else None
    self.pid = torch.empty(hint_N, dtype=torch.long)
    self.age = torch.empty(hint_N, 1)
    self.sex = torch.empty(hint_N, 1)
    self.dx = torch.empty(hint_N, num_classes)
    self.n_removed = 0

    r = 0
    for i, entry in tqdm(enumerate(df.iloc)):
      age = entry["Age"]
      sex = entry["Gender_Male"]
      if not np.isfinite(age) or not np.isfinite(sex) or (sex != 0 and sex != 1): continue
      
      recording = PhysionetDataset._read_recording(entry["Patient"], rdim)
      if recording is None: continue
      
      dx = torch.tensor(df.iloc[i][classes].astype(np.int).to_numpy()).unsqueeze(0)
      N = len(recording)
      while r + N > self.orig.shape[0]: self._grow(replace)

      if self.orig: self.orig[r:r+N] = recording
      self.pid[r:r+N] = i
      self.age[r:r+N] = age
      self.sex[r:r+N] = sex
      self.dx[r:r+N] = dx

      r += N
    self._trim(r)
    self.age = (self.age - self.age.min()) / (self.age.max() - self.age.min())
    if frac and frac_mode: self._fractionate(frac, frac_mode)

  def __getitem__(self, i):
    #pid keeps track of which crops came from which patients in case we need it
    return self.orig[i], self.pid[i], self.age[i], self.sex[i], self.dx[i]
  
  def __len__(self):
    return len(self.orig)
  
  def _grow(self, exclude_orig: bool):
    if not exclude_orig: self.orig = torch.cat((self.orig, torch.empty_like(self.orig)), dim=0)
    self.pid = torch.cat((self.pid, torch.empty_like(self.pid)), dim=0)
    self.age = torch.cat((self.age, torch.empty_like(self.age)), dim=0)
    self.sex = torch.cat((self.sex, torch.empty_like(self.sex)), dim=0)
    self.dx  = torch.cat((self.dx , torch.empty_like(self.dx)) , dim=0)

  def _trim(self, r: int):
    self._apply_mask(slice(r))
  
  def _apply_mask(self, mask):
    self.orig = self.orig[mask].contiguous()
    self.pid  = self.pid[mask].contiguous()
    self.age  = self.age[mask].contiguous()
    self.sex  = self.sex[mask].contiguous()
    self.dx   = self.dx[mask].contiguous()
  
  def _fractionate(self, frac: float, frac_mode: str):
    mask = torch.zeros(len(self), dtype=torch.bool)
    if frac >= 1: return
    elif frac_mode == "full_rand": mask = torch.rand(len(self)) < frac
    elif frac_mode == "min_rand":
      pid_sorted = torch.histc(self.pid, min=0, bins=self.pid.max()).argsort()
      required = torch.zeros(self.dx.shape[1], dtype=torch.bool)
      for pid in pid_sorted:
        if required.all(): break
        relevant = self.pid == pid

        dx = self.dx[relevant].bool()
        assert (dx == dx[0]).all(), "Incorrect pid mask"
        dx = dx[0]

        if dx[dx ^ required].any():
          mask |= relevant
          required |= dx
      if (mask.count_nonzero() / len(mask)) < frac: mask[torch.rand_like(mask) < frac] = True
    else: raise ValueError(f"Unimplemented fractioning method: {frac_mode}")
    self.n_removed = (~mask).count_nonzero()
    self._apply_mask(mask)

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
  
  def clear(self):
    del self.orig
    del self.pid
    del self.age
    del self.sex
    del self.dx