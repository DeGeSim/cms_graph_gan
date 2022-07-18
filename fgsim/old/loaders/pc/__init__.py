"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

if "pytest" not in sys.modules:

    import torch

    from fgsim.io import batch_tools
    from fgsim.io.loader import LoaderInfo
    from fgsim.loaders.pcgraph.files import files, len_dict, save_len_dict
    from fgsim.loaders.pcgraph.scaler import scaler

    from .seq import postprocess_switch, process_seq

    loader = LoaderInfo(
        process_seq=process_seq,
        Batch=torch.Tensor,
        files=files,
        len_dict=len_dict,
        postprocess_switch=postprocess_switch,
        batch_tools=batch_tools,
    )
