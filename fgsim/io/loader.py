from dataclasses import dataclass
from typing import List, Type, Union

from queueflow import StepBase
from torch.multiprocessing import Queue, Value

from .file_manager import FileManager
from .scaler_base import ScalerBase


@dataclass
class Loader:
    file_manager: FileManager
    scaler: ScalerBase
    process_seq: List[Union[StepBase, Queue]]
    shared_postprocess_switch: Value
    shared_batch_size: Value
    Batch: Type
