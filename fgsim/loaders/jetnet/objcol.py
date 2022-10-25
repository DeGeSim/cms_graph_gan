from pathlib import Path
from typing import List, Tuple

import h5py
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch_geometric.data import Data

from fgsim.io import FileManager, ScalerBase


def path_to_len(fn: Path) -> int:
    with h5py.File(fn, "r") as f:
        res = f["particle_features"].shape[0]
    return res


file_manager = FileManager(path_to_len)


def readpath(
    fn: Path,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(fn, "r") as f:
        res = (
            torch.tensor(f["jet_features"][start:end]),
            torch.tensor(f["particle_features"][start:end]),
        )
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


def contruct_graph_from_row(chk: Tuple[torch.Tensor, torch.Tensor]) -> Data:
    y, x = chk
    res = Data(
        x=x[x[..., 3].bool(), :3].reshape(-1, 3),
        y=y.reshape(1, -1),
    )
    return res


scaler = ScalerBase(
    files=file_manager.files,
    len_dict=file_manager.file_len_dict,
    transfs=[
        StandardScaler(),
        StandardScaler(),
        PowerTransformer(method="box-cox", standardize=True),
    ],
    read_chunk=read_chunks,
    transform_wo_scaling=contruct_graph_from_row,
)
