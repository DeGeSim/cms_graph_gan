"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys


def loadfiles():
    # Load files
    ds_path = Path(conf.path.dataset)
    assert ds_path.is_dir()
    files = sorted(ds_path.glob(conf.loader.dataset_glob))
    if len(files) < 1:
        raise RuntimeError("No datasets found")

    import h5py as h5
    import pandas as pd

    # load lengths
    if not os.path.isfile(conf.path.ds_lenghts):
        len_dict = {}
        for fn in files:
            h5_file = pd.read_csv(fn)
            len_dict[str(fn)] = len(h5_file)
        ds_processed = Path(conf.path.dataset_processed)
        if not ds_processed.is_dir():
            ds_processed.mkdir()
        with open(conf.path.ds_lenghts, "w") as f:
            yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
    else:
        with open(conf.path.ds_lenghts, "r") as f:
            len_dict = yaml.load(f, Loader=yaml.SafeLoader)
    return files, len_dict


if "pytest" not in sys.modules:
    import os
    from pathlib import Path
    from typing import List, Tuple

    import joblib
    import uproot
    import yaml
    from torch_geometric.data import Batch

    from fgsim.config import conf
    from fgsim.io import batch_tools
    from fgsim.io.loader import Loader

    from .seq import postprocess_switch, process_seq

    files, len_dict = loadfiles()
    # Scaling
    # make sure there is a scaler:

    loader = Loader(
        process_seq=process_seq,
        Batch=Batch,
        files=files,
        len_dict=len_dict,
        postprocess_switch=postprocess_switch,
        batch_tools=batch_tools,
    )
