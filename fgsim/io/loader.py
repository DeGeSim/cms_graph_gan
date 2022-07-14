from dataclasses import dataclass
from typing import Type

from queueflow import Sequence
from torch.multiprocessing import Value

from .file_manager import FileManager
from .scaler_base import ScalerBase


@dataclass
class Loader:
    file_manager: FileManager
    scaler: ScalerBase
    process_seq: Sequence
    postprocess_switch: Value
    Batch: Type
