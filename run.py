#!/usr/bin/env python
__author__ = "Bryan Gopal"

from attrdict import AttrDict
from dataprocessing import PhysionetDataModule, GANDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from systems import ECGAN, Classifier
from utils import configs

def main():
  seed_everything(6)
     
  pdm = PhysionetDataModule(**configs.pdm)
  gan = train(pdm)
  gan.enable_downstream()
  gdm = GANDataModule(gan, *pdm.get_labels(), **configs.gdm)
  run_downstream(gdm)


def train(dm: PhysionetDataModule) -> ECGAN:
  cfg = configs.gan
  model = ECGAN(cfg.hparams)
  if cfg.skip: return ECGAN.load_from_checkpoint(cfg.path) if cfg.path else model
  
  trainer = Trainer(**cfg.trainer, resume_from_checkpoint=cfg.path)
  trainer.tune(model, datamodule=dm)
  trainer.fit(model, datamodule=dm)
  
  return model

def run_downstream(dm: GANDataModule):
  cfg = configs.ds
  if cfg.skip: return

  callbacks = _get_callbacks(cfg)

  classifier = Classifier(cfg.hparams)
  
  trainer = Trainer(callbacks=callbacks, **cfg.trainer, resume_from_checkpoint=cfg.path)
  trainer.tune(classifier, datamodule=dm)
  trainer.fit(classifier, datamodule=dm)
  trainer.test(ckpt_path="best", datamodule=dm, verbose=False)

def _get_callbacks(config: AttrDict):
  m = config.metric
  return [
    EarlyStopping(  **m, patience=30),
    ModelCheckpoint(**m, save_top_k=1, filename=f"{{epoch}}-{{{m.monitor}:.4f}}")
  ]

if __name__ == "__main__":
  main()