from .parsing import args
import os

general_path = os.path.expanduser("~/ecgan-runs")
data_path = f"{general_path}/data"
raw_data_path = f"{data_path}/raw"

main_dir = args.ds_gen_mode if args.ds_gen_mode else "gan"
check_dir = f"{general_path}/checkpoints/{args.frac}/{main_dir}"