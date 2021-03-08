#!/usr/bin/env python
__author__ = "Bryan Gopal"

from pytorch_lightning import seed_everything
from systems import ECGAN
import torch
from utils import configs, classes, check_dir

def main():
  seed_everything(6)
  cfg = configs.gan
  if not cfg.path: raise ValueError("Sample Generation Requires a Saved GAN.")

  gan = ECGAN.load_from_checkpoint(cfg.path)
  age = torch.tensor([[0.5]])
  sex = torch.tensor([[0]])
  dx  = torch.from_numpy(classes == "164889003").float().unsqueeze_(0) #atrial fibrilation

  torch.save(gan(age, sex, dx), f"{check_dir}/58_F_AFIB.pt")

if __name__ == "__main__":
  main()