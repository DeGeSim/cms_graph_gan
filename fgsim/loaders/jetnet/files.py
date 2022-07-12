import os
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from fgsim.config import conf


def get_file_list():
    # Load files
    ds_path = Path(conf.path.dataset)
    assert ds_path.is_dir()
    files = sorted(ds_path.glob(conf.loader.dataset_glob))
    if len(files) < 1:
        raise RuntimeError("No datasets found")
    return [f for f in files]


files = get_file_list()


def save_len_dict():
    len_dict = {}
    for fn in files:
        h5_file = pd.read_csv(fn)
        len_dict[str(fn)] = len(h5_file)
    ds_processed = Path(conf.path.dataset_processed)
    if not ds_processed.is_dir():
        ds_processed.mkdir()
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)


def load_files() -> Dict[Path, int]:
    # load lengths
    if not os.path.isfile(conf.path.ds_lenghts):
        save_len_dict()
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)
    # convert to path
    len_dict = {Path(k): v for k, v in len_dict.items()}
    return len_dict


len_dict = load_files()
