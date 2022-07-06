"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

if "pytest" not in sys.modules:

    from torch_geometric.data import Batch

    from fgsim.io import batch_tools
    from fgsim.io.loader import Loader

    from .files import files, len_dict
    from .scaler import scaler
    from .seq import postprocess_switch, process_seq

    loader = Loader(
        process_seq=process_seq,
        Batch=Batch,
        files=files,
        len_dict=len_dict,
        postprocess_switch=postprocess_switch,
        batch_tools=batch_tools,
    )
