#!/usr/bin/env python
__author__ = "Bryan Gopal"

from argparse import ArgumentParser, Namespace
from pytorch_lightning import seed_everything
import os
from systems import ECGAN
import torch
import numpy as np

classes = np.array(sorted(("270492004", "164889003", "164890007", "426627000", 
                           "713427006", "713426002", "445118002", "39732003" , 
                           "164909002", "251146004", "698252002", "10370003" , 
                           "284470004", "427172004", "164947007", "111975006", 
                           "164917005", "47665007" , "427393009", "426177001", 
                           "426783006", "427084000", "164934002", "59931005"  )))

def main(args: Namespace):
  seed_everything(6)

  gan = ECGAN.load_from_checkpoint(args.path)
  age = torch.tensor([[0.5]])
  sex = torch.tensor([[0]])
  dx  = torch.from_numpy(classes == "426783006").float().unsqueeze_(0)

  torch.save(gan(age, sex, dx), f"{os.path.split(args.path)[0]}/58_F_SINUS.pt")

if __name__ == "__main__":
  _parser = ArgumentParser()

  _parser.add_argument("path", type=os.path.expanduser)
  main(_parser.parse_args())