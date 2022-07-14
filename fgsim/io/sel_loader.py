import importlib
from typing import List, Type

from fgsim.config import conf
from fgsim.io.loader import Loader

# Import the specified processing sequence
loader: Loader = importlib.import_module(f"fgsim.loaders.{conf.loader_name}").loader

file_manager = loader.file_manager
scaler = loader.scaler
process_seq = loader.process_seq
postprocess_switch = loader.postprocess_switch
Batch: Type = loader.Batch

DataSetType = List[Batch]
files = loader.file_manager.files
len_dict = loader.file_manager.file_len_dict
