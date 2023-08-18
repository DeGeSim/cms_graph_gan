from pathlib import Path
from typing import List, Tuple

import torch
from jetnet.datasets import JetNet
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from torch_geometric.data import Data
from torch_scatter import scatter_add

from fgsim.config import conf
from fgsim.io import FileManager, ScalerBase
from fgsim.io.dequantscaler import dequant_stdscale
from fgsim.utils import check_tensor

jn_dict = {}


def get_jn(fn):
    fn = str(fn)

    if fn not in jn_dict:
        jn_dict[fn] = JetNet.getData(
            jet_type=fn,
            data_dir=Path(conf.loader.dataset_path).expanduser(),
            num_particles=conf.loader.n_points,
        )
    return jn_dict[fn]


def path_to_len(fn: Path) -> int:
    return get_jn(fn)[0].shape[0]


file_manager = FileManager(path_to_len, files=[Path(conf.loader.jettype)])


def readpath(
    fn: Path,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    particle_data, jet_data = get_jn(fn)
    res = (
        torch.tensor(jet_data[start:end], dtype=torch.float),
        torch.tensor(particle_data[start:end], dtype=torch.float),
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
    transfs_x=[
        StandardScaler(),
        StandardScaler(),
        PowerTransformer(method="box-cox", standardize=True),
    ],
    transfs_y=[
        MinMaxScaler((-1, 1)),  # type
        StandardScaler(),  # pt
        StandardScaler(),  # eta
        StandardScaler(),  # mass
        dequant_stdscale((0, conf.loader.n_points + 1)),  # num_particles
    ],
    read_chunk=read_chunks,
    transform_wo_scaling=contruct_graph_from_row,
)


def norm_pt_sum(pts, batchidx):
    pt_scaler = scaler.transfs_x[2]

    assert pt_scaler.method == "box-cox"
    assert pt_scaler.standardize
    # get parameters for the backward tranformation
    lmbd = pt_scaler.lambdas_[0]
    mean = pt_scaler._scaler.mean_[0]
    scale = pt_scaler._scaler.scale_[0]

    # Backwards transform
    pts = pts.clone().double() * scale + mean
    check_tensor(pts)
    if lmbd == 0:
        pts = torch.exp(pts.clone())
    else:
        pts = torch.pow(pts.clone() * lmbd + 1, 1 / lmbd)
    check_tensor(pts)

    # Norm
    ptsum_per_batch = scatter_add(pts, batchidx, dim=-1)
    pts = pts / ptsum_per_batch[batchidx]
    check_tensor(pts)

    # Forward transform
    if lmbd == 0:
        pts = torch.log(pts.clone())
    else:
        pts = (torch.pow(pts.clone(), lmbd) - 1) / lmbd

    pts = (pts.clone() - mean) / scale
    check_tensor(pts)
    return pts.float()
