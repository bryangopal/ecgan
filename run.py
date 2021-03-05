#!/usr/bin/env python
__author__ = "Bryan Gopal"

from attrdict import AttrDict
from dataprocessing import PhysionetDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from systems import ECGAN, Classifier
from utils import configs

def main():
  seed_everything(6)
  pdm = PhysionetDataModule(**configs.pdm)
  train(pdm)
  run_downstream(pdm)

def train(pdm: PhysionetDataModule) -> ECGAN:
  cfg = configs.gan
  model = ECGAN(cfg.hparams)
  if cfg.skip: 
    configs.ds.hparams.gan_path = cfg.path
    return
  
  trainer = Trainer(**cfg.trainer, resume_from_checkpoint=cfg.path)
  trainer.tune(model, datamodule=pdm)
  trainer.fit(model, datamodule=pdm)
  
  configs.ds.hparams.gan_path = trainer.checkpoint_callback.last_model_path

def run_downstream(pdm: PhysionetDataModule):
  cfg = configs.ds
  if cfg.skip: return

  callbacks = _get_callbacks(cfg)

  classifier = Classifier(cfg.hparams)
  
  trainer = Trainer(callbacks=callbacks, **cfg.trainer, resume_from_checkpoint=cfg.path)
  trainer.tune(classifier, datamodule=pdm)
  trainer.fit(classifier, datamodule=pdm)
  trainer.test(ckpt_path="best", datamodule=pdm, verbose=False)

def _get_callbacks(config: AttrDict):
  m = config.metric
  return [
    EarlyStopping(  **m, patience=30),
    ModelCheckpoint(**m, save_top_k=1, filename=f"{{epoch}}-{{{m.monitor}:.4f}}")
  ]

if __name__ == "__main__":
  main()