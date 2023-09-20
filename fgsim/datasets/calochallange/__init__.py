"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

import caloutils

from fgsim.config import conf

from .postprocess import postprocess

if "dataset_2" in conf.loader.dataset_glob:
    caloutils.init_calorimeter("cc_ds2")
elif "dataset_3" in conf.loader.dataset_glob:
    caloutils.init_calorimeter("cc_ds3")
else:
    raise Exception("No such dataset")

if "pytest" not in sys.modules:
    from torch_geometric.data import Batch

    from fgsim.io import LoaderInfo

    from .objcol import file_manager, scaler
    from .seq import process_seq, shared_batch_size, shared_postprocess_switch

    loader = LoaderInfo(
        file_manager=file_manager,
        scaler=scaler,
        process_seq=process_seq,
        shared_postprocess_switch=shared_postprocess_switch,
        shared_batch_size=shared_batch_size,
        Batch=Batch,
    )
