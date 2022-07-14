from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch_geometric.data import Data

from fgsim.io import FileManager, ScalerBase


def path_to_len(fn: Path) -> int:
    return len(
        pd.read_csv(
            fn,
            sep=" ",
            header=None,
        )
    )


file_manager = FileManager(path_to_len)


def readpath(
    fn: Path,
    start: Optional[int],
    end: Optional[int],
) -> pd.DataFrame:
    if start is end is None:
        return pd.read_csv(
            fn,
            sep=" ",
            header=None,
        )
    elif isinstance(start, int) and isinstance(end, int):
        return pd.read_csv(
            fn,
            skiprows=start,
            nrows=(end - start),
            sep=" ",
            header=None,
        )
    else:
        raise ValueError()


def read_chunks(chunks: List[Tuple[Path, int, int]]) -> torch.Tensor:
    chunks_list = []
    for chunk in chunks:
        chunks_list.append(readpath(*chunk))
    res = pd.concat(chunks_list).values.reshape(-1, 30, 4)
    # res = res[..., :3]
    return torch.tensor(res).float()


def contruct_graph_from_row(row) -> Data:
    res = Data(x=row[:, :3], mask=row[:, :1].reshape(-1).bool())
    res.x[~res.mask] = -5.0
    return res


scaler = ScalerBase(
    file_manager.files,
    file_manager.file_len_dict,
    [
        StandardScaler(),
        StandardScaler(),
        PowerTransformer(method="box-cox", standardize=True),
    ],
    read_chunks,
    contruct_graph_from_row,
)
