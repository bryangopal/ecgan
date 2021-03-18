#!/usr/bin/env python
__author__ = "Bryan Gopal"

from argparse import ArgumentParser, Namespace
from dataprocessing.datamodule import PhysionetDataModule
from torch.nn import functional as F
from torch import Tensor
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional import auroc
import os
from systems import Classifier
import torch
import numpy as np
from utils import configs, num_classes, class_labels, classes

def multiclass_auroc(input: Tensor, target: Tensor, print_classes: bool = False) -> Tensor:
  with torch.no_grad():
    pred = F.softmax(input, dim=1)

    aurocs = {}
    for c in range(num_classes):
      if 1 in target[:, c] and 0 in target[:, c]:
        aurocs[class_labels[classes[c]]] = auroc(pred[:, c], target[:, c])

    if not aurocs: raise ValueError("No positive labels found in target")
    if print_classes: print(aurocs)
    return torch.mean(torch.tensor(tuple(aurocs.values())))

def main():
  seed_everything(6)

  classifier = Classifier.load_from_checkpoint(configs.ds.path)
  pdm = PhysionetDataModule(**configs.pdm)
  pdm.setup("test")

  input = torch.empty(0)
  target = torch.empty(0)
  for x, *_, y in pdm.test_dataloader():
    with torch.no_grad():
      input = torch.cat((input, classifier(x)), dim=0)
      target = torch.cat((target, y))
  
  multiclass_auroc(input, target, True)

  

if __name__ == "__main__":
  main()