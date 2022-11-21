import importlib
from typing import List, Type

from fgsim.config import conf
from fgsim.io.loader import LoaderInfo

# Import the specified processing sequence
loader_info: LoaderInfo = importlib.import_module(
    f"fgsim.loaders.{conf.dataset_name}"
).loader

file_manager = loader_info.file_manager
scaler = loader_info.scaler
process_seq = loader_info.process_seq
shared_postprocess_switch = loader_info.shared_postprocess_switch
Batch: Type = loader_info.Batch

DataSetType = List[Batch]
files = loader_info.file_manager.files
len_dict = loader_info.file_manager.file_len_dict
