import importlib
from typing import List

from fgsim.config import conf
from fgsim.io.loader import Loader

# Import the specified processing sequence
loader: Loader = importlib.import_module(
    f"fgsim.io.{conf.loader.qf_seq_name}"
).loader

Batch = loader.Batch
DataSetType = List[Batch]
batch_tools = loader.batch_tools

process_seq = loader.process_seq
files = loader.files
len_dict = loader.len_dict
postprocess_switch = loader.postprocess_switch
