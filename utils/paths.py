from .parsing import args

import os
from shutil import rmtree
from typing import List

general_path = os.path.expanduser("~/ecgan-runs")
data_path = f"{general_path}/data"
saved_data_path = f"{data_path}/saved"
raw_data_path = f"{data_path}/raw"
check_dir = f"{general_path}/checkpoints/{args.dir}"

if args.clear_data and os.path.exists(saved_data_path): rmtree(saved_data_path)
os.makedirs(saved_data_path, exist_ok=True)

def all_paths_exist(*paths: List[str]):
  for path in paths:
    if not os.path.exists(path): return False
  return True