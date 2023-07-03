from pathlib import Path
from typing import List, Tuple

import h5py
import torch
from sklearn.preprocessing import PowerTransformer
from torch_geometric.data import Data

from fgsim.config import conf
from fgsim.io import FileManager, ScalerBase
from fgsim.io.dequantscaler import dequant_stdscale


def path_to_len(fn: Path) -> int:
    return len(h5py.File(fn, "r")["incident_energies"])


file_manager = FileManager(
    path_to_len,
    files=list(
        Path(conf.loader.dataset_path).expanduser().glob(conf.loader.dataset_glob)
    ),
)


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


def contruct_graph_from_row(chk: Tuple[torch.Tensor, torch.Tensor]) -> Data:
    E, shower = chk[0].clone(), chk[1].clone()
    num_z = 45
    num_alpha = 16
    num_r = 9
    shower = shower.reshape(num_z, num_alpha, num_r)
    idxs = torch.where(shower)
    h_energy = shower[idxs]
    z, alpha, r = idxs

    pc = torch.stack([h_energy, z, alpha, r]).T
    assert not pc.isnan().any()
    num_hits = torch.tensor(idxs[0].shape).float()
    res = Data(x=pc, y=torch.concat([E, num_hits]).reshape(1, 2))
    return res


scaler = ScalerBase(
    files=file_manager.files,
    len_dict=file_manager.file_len_dict,
    transfs_x=[
        PowerTransformer(method="box-cox", standardize=True),
        dequant_stdscale(),
        dequant_stdscale(),
        dequant_stdscale(),
    ],
    transfs_y=[
        PowerTransformer(method="box-cox", standardize=True),  # Energy
        dequant_stdscale((0, conf.loader.n_points + 1)),  # num_particles
    ],
    read_chunk=read_chunks,
    transform_wo_scaling=contruct_graph_from_row,
)
