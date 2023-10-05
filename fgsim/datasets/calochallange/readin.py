from pathlib import Path
from typing import List, Tuple

import h5py
import torch

from fgsim.config import conf
from fgsim.io import FileManager


def readpath(
    fn: Path,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(fn, "r") as electron_file:
        energies = electron_file["incident_energies"][start:end]
        showers = electron_file["showers"][start:end]
    res = (torch.Tensor(energies), torch.Tensor(showers))
    return res


def read_chunks(
    chunks: List[Tuple[Path, int, int]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    chunks_list = []
    for chunk in chunks:
        chunks_list.append(readpath(*chunk))
    res = (
        torch.concat([e[0] for e in chunks_list]),
        torch.concat([e[1] for e in chunks_list]),
    )
    return [(res[0][ievent], res[1][ievent]) for ievent in range(len(res[1]))]


def path_to_len(fn: Path) -> int:
    return len(h5py.File(fn, "r")["incident_energies"])


file_manager = FileManager(
    path_to_len,
    files=list(
        Path(conf.loader.dataset_path).expanduser().glob(conf.loader.dataset_glob)
    ),
)
