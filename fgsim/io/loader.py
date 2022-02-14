from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

from queueflow import Sequence
from torch.multiprocessing import Value


@dataclass
class Loader:
    process_seq: Sequence
    Batch: type
    files: List[Path]
    len_dict: Dict[str, int]
    postprocess_switch: Value
    batch_tools: Optional[ModuleType]
