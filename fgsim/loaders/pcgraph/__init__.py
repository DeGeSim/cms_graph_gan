"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys


def get_file_list():
    # Load files
    ds_path = Path(conf.path.dataset)
    assert ds_path.is_dir()
    files = sorted(ds_path.glob(conf.loader.dataset_glob))
    if len(files) < 1:
        raise RuntimeError("No datasets found")
    return files


def loadfiles():
    files = get_file_list()
    # load lengths
    if not os.path.isfile(conf.path.ds_lenghts):
        len_dict = {}
        for fn in files:
            with uproot.open(fn) as rfile:
                len_dict[str(fn)] = rfile[conf.loader.rootprefix].num_entries
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

    import uproot
    import yaml

    from fgsim.config import conf
    from fgsim.io import batch_tools
    from fgsim.io.loader import Loader

    from .event import Batch
    from .scaler import comb_transf, load_scaler
    from .scaler import save_scaler as _save_scaler
    from .seq import postprocess_switch, process_seq

    files, len_dict = loadfiles()

    def save_scaler():
        return _save_scaler(files, len_dict)

    loader = Loader(
        process_seq=process_seq,
        Batch=Batch,
        files=files,
        len_dict=len_dict,
        postprocess_switch=postprocess_switch,
        batch_tools=batch_tools,
    )
