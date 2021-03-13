from .parsing import args

import os
from shutil import rmtree
from typing import List

general_path = os.path.expanduser("~/ecgan-runs")
data_path = f"{general_path}/data"
raw_data_path = f"{data_path}/raw"

main_dir = args.gen_mode if args.gen_mode else "gan"
check_dir = f"{general_path}/checkpoints/{args.frac}/{main_dir}"