"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import sys

from fgsim.io.loader import Loader

if "pytest" not in sys.modules:
    import os
    from collections import defaultdict
    from pathlib import Path

    import uproot
    import yaml
    from torch_geometric.data import Batch

    # from torch_geometric.data.batch import DataBatch as Batch
    from fgsim.config import conf
    from fgsim.io import batch_tools

    from .seq import postprocess_switch, process_seq

    files = list(range(20))
    len_dict = defaultdict(lambda: conf.loader.events_per_file)

    loader = Loader(
        process_seq=process_seq,
        Batch=Batch,
        files=files,
        len_dict=len_dict,
        postprocess_switch=postprocess_switch,
        batch_tools=batch_tools,
    )
