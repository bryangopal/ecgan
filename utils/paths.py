from .parsing import args
import os

general_path = os.path.expanduser("~/ecgan-runs")
data_path = f"{general_path}/data"
raw_data_path = f"{data_path}/raw"

main_dir = args.ds_gen_mode if args.ds_gen_mode else "gan"
co_dir = f"-{args.ds_gen_frac}" if args.ds_gen_mode == "augment" else ""
check_dir = f"{general_path}/checkpoints/{args.gan_data_frac}/{main_dir}{co_dir}"