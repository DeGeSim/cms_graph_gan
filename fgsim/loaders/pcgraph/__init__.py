"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

if "pytest" not in sys.modules:

    from torch_geometric.data import Batch

    from fgsim.io.loader import Loader
    from fgsim.loaders.hgcal.objcol import file_manager, scaler
    from fgsim.loaders.hgcal.seq import process_seq, shared_postprocess_switch

    loader = Loader(
        file_manager=file_manager,
        scaler=scaler,
        process_seq=process_seq,
        shared_postprocess_switch=shared_postprocess_switch,
        Batch=Batch,
    )
