from argparse import ArgumentParser

_parser = ArgumentParser()

_parser.add_argument("dir", type=str)
_parser.add_argument("frac", type=float)

_parser.add_argument("--gen_mode", type=str, default=None,
                     choices=["replace", "augment"])  
_parser.add_argument("--skip_gan", action="store_true")
_parser.add_argument("--gan_path", default=None, type=str)
_parser.add_argument("--gan_lr", default=2e-4, type=float)
_parser.add_argument("--z_dim", default=256, type=int)

_parser.add_argument("--skip_ds", action="store_true")
_parser.add_argument("--ds_path", default=None, type=str)
_parser.add_argument("--ds_encoder", type=str, default="resnet18",
                     choices=["resnet50", "resnet18"])   
_parser.add_argument("--ds_lr", default=2e-4, type=float)

_parser.add_argument("--fast_dev_run", action="store_true")
_parser.add_argument("--crop_time", type=int, default=5)
_parser.add_argument("--batch_size", type=int, default=2048)

args = _parser.parse_args()

if args.skip_gan and args.gen_mode == "augment":
  args.batch_size = int(args.batch_size * args.frac) 