"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

if "pytest" not in sys.modules:
    import os
    from pathlib import Path
    from typing import List, Tuple

    import uproot
    import yaml

    from fgsim.config import conf

    from .event import Batch, Event
    from .seq import postprocess_switch, process_seq

    # Load files
    ds_path = Path(conf.path.dataset)
    assert ds_path.is_dir()
    files = sorted(ds_path.glob(conf.loader.dataset_glob))
    if len(files) < 1:
        raise RuntimeError("No datasets found")

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
