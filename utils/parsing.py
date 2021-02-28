from argparse import ArgumentParser

_parser = ArgumentParser()

_parser.add_argument("dir", type=str)

_parser.add_argument("--skip_gan", action="store_true")
_parser.add_argument("--gan_path", default=None, type=str)
_parser.add_argument("--gan_lr", default=2e-4, type=float)
_parser.add_argument("--z_dim", default=256, type=int)

_parser.add_argument("--skip_ds", action="store_true")
_parser.add_argument("--ds_path", default=None, type=str)
_parser.add_argument("--ds_encoder", type=str, default="resnet50",
                     choices=["resnet50", "resnet18_bn"])   
_parser.add_argument("--ds_lr", default=2e-4, type=float)
_parser.add_argument("--single_lead", action="store_true")

_parser.add_argument("--fast_dev_run", action="store_true")
_parser.add_argument("--crop_time", type=int, default=5)
_parser.add_argument("--batch_size", type=float, default=1024)

args = _parser.parse_args()