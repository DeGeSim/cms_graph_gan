from pathlib import Path
from typing import List, Tuple

from sklearn.preprocessing import StandardScaler

from fgsim.io import FileManager, ScalerBase

from .transform import transform


def read_chunks(chunks: List[Tuple[Path, int, int]]) -> List[int]:
    return [1 for chunk in chunks for _ in range(chunk[2] - chunk[1])]


def path_to_len(_) -> int:
    return 500_000


file_manager = FileManager(path_to_len=path_to_len)

scaler = ScalerBase(
    file_manager.files,
    file_manager.file_len_dict,
    read_chunks,
    transform,
    transfs=[StandardScaler(), StandardScaler()],
)
