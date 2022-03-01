"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import sys

from fgsim.io.loader import Loader

if "pytest" not in sys.modules:
    import os
    from pathlib import Path

    import uproot
    import yaml
    from torch_geometric.data import Batch

    # from torch_geometric.data.batch import DataBatch as Batch
    from fgsim.config import conf
    from fgsim.io import batch_tools

    from .seq import postprocess_switch, process_seq

    # Load files
    ds_path = Path(conf.path.dataset)
    assert ds_path.is_dir()
    files = sorted(ds_path.glob(conf.path.dataset_glob))
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

    loader = Loader(
        process_seq=process_seq,
        Batch=Batch,
        files=files,
        len_dict=len_dict,
        postprocess_switch=postprocess_switch,
        batch_tools=batch_tools,
    )
