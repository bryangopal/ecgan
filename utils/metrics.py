from .data_info import classes, num_classes, class_labels

import numpy as np
from pytorch_lightning.metrics.functional import auroc
import torch
from torch import Tensor
from torch.nn import functional as F

def multiclass_auroc(input: Tensor, target: Tensor, print_classes: bool = False) -> Tensor:
  _check_metric_inputs(input, target)
  with torch.no_grad():
    pred = F.sigmoid(input, dim=1)

    aurocs = {}
    for c in range(num_classes):
      if 1 in target[:, c] and 0 in target[:, c]:
        aurocs[class_labels[classes[c]]] = auroc(pred[:, c], target[:, c])
    
    if not aurocs: raise ValueError("No positive labels found in target")
    if print_classes: print(aurocs)
    return torch.mean(torch.tensor(tuple(aurocs.values())))

def _check_metric_inputs(input: Tensor, target: Tensor):
  if not torch.all(torch.isfinite(input)): 
    raise ValueError("Input tensor passed into a metric contains nonfinite values, "
                    f"inf={torch.count_nonzero(torch.isinf(input))}, "
                    f"nan={torch.count_nonzero(torch.isnan(input))}")
  if input.shape != target.shape or target.shape[1] != num_classes or len(target.shape) != 2:
    raise ValueError(f"Malformed args, input: {input.shape}, target: {target.shape},"
                     f"num_classes: {num_classes}")