__author__ = "Bryan Gopal"
from .data_info import *
from .metrics import *
from .parsing import *
from .paths import *

from attrdict import AttrDict
import os
from psutil import cpu_count

_trainer = {
  "gpus": 1,
  "sync_batchnorm": True, 
  "default_root_dir": check_dir,
  "deterministic": True,
  "fast_dev_run": args.fast_dev_run,
  "terminate_on_nan": True,
  "enable_pl_optimizer": True,
  "max_epochs": 250,
  "auto_scale_batch_size": True,
  "precision": 16
}

_expand_path = lambda path: os.path.expanduser(path) if path else None
configs = AttrDict({
  "gan": {
    "skip": args.skip_gan,
    "path": _expand_path(args.gan_path),
    "hparams": {
      "lr": args.gan_lr,
      "z_dim": args.z_dim,
      "num_classes": num_classes,
      "C": num_leads,
      "L": rL,
    },
    "trainer": _trainer
  },

  "ds": {
    "skip": args.skip_ds,
    "path": _expand_path(args.ds_path),
    "metric": {
      "monitor": "le_val_auroc",
      "mode": "max"
    },
    "hparams": {
      "encoder": args.ds_encoder,
      "lr": args.ds_lr,
      "num_channels": num_leads,
      "num_classes": num_classes
    },
    "trainer": {
      **_trainer,
      "auto_lr_find": True
    }
  },

  "pdm": {
    "path": data_path,
    "folds": {
      "train": [0, 1, 4, 5, 6, 7, 8, 9],
      "val": [2],
      "test": [3]
    },
    "dims": (num_leads, rL),
    "batch_size": 128,
    "num_workers": cpu_count()
  },

  "gdm": {
    "batch_size": 128,
    "num_workers": cpu_count()
  },
})
