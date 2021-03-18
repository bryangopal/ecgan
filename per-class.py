#!/usr/bin/env python
__author__ = "Bryan Gopal"

from argparse import ArgumentParser, Namespace
from dataprocessing.datamodule import PhysionetDataModule
from torch.nn import functional as F
from torch import Tensor
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics import AUROC
import os
from systems import Classifier
import torch
import numpy as np
from utils import configs, num_classes, class_labels, classes
def main():
  seed_everything(6)

  classifier = Classifier.load_from_checkpoint(configs.ds.path)
  auroc = AUROC(num_classes, compute_on_step=False, average=None)
  pdm = PhysionetDataModule(**configs.pdm)
  pdm.setup("test")

  for x, *_, y in pdm.test_dataloader():
    with torch.no_grad():
     auroc(F.softmax(classifier(x), dim=-1), y.long())
  
  x = auroc.compute()
  aucs = {}
  for c in range(num_classes):
    aucs[class_labels[classes[c]]] = x[c]
  
  print(aucs)
  

if __name__ == "__main__":
  main()