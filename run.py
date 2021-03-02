#!/usr/bin/env python
__author__ = "Bryan Gopal"

from attrdict import AttrDict
from dataprocessing import PhysionetDataModule, DownstreamDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from systems import ECGAN, Classifier
from utils import configs

def main():
  seed_everything(6)
  run_downstream(*train())

def train() -> ECGAN:
  cfg = configs.gan
  model = ECGAN(cfg.hparams)
  if cfg.skip: return ECGAN.load_from_checkpoint(cfg.path) if cfg.path else model
  
  pdm = PhysionetDataModule(**configs.pdm)
  trainer = Trainer(**cfg.trainer, resume_from_checkpoint=cfg.path)
  trainer.tune(model, datamodule=pdm)
  trainer.fit(model, datamodule=pdm)
  
  model.enable_downstream()
  return model, pdm

def run_downstream(gan: ECGAN, pdm: PhysionetDataModule):
  cfg = configs.ds
  if cfg.skip: return

  gdm = DownstreamDataModule(gan, pdm, **configs.gdm)
  callbacks = _get_callbacks(cfg)

  classifier = Classifier(cfg.hparams)
  
  trainer = Trainer(callbacks=callbacks, **cfg.trainer, resume_from_checkpoint=cfg.path)
  trainer.tune(classifier, datamodule=gdm)
  trainer.fit(classifier, datamodule=gdm)
  trainer.test(ckpt_path="best", datamodule=gdm, verbose=False)

def _get_callbacks(config: AttrDict):
  m = config.metric
  return [
    EarlyStopping(  **m, patience=30),
    ModelCheckpoint(**m, save_top_k=1, filename=f"{{epoch}}-{{{m.monitor}:.4f}}")
  ]

if __name__ == "__main__":
  main()