"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

if "pytest" not in sys.modules:
    from torch_geometric.data import Batch

    from fgsim.io import Loader

    from .objcol import file_manager, scaler
    from .seq import postprocess_switch, process_seq

    loader = Loader(
        file_manager=file_manager,
        scaler=scaler,
        process_seq=process_seq,
        postprocess_switch=postprocess_switch,
        Batch=Batch,
    )
