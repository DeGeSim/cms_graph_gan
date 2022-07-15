from pathlib import Path
from typing import List, Optional, Tuple

import awkward as ak
import uproot
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

from fgsim.config import conf
from fgsim.io import FileManager, ScalerBase

from .transform import hitlist_to_graph


def readpath(
    fn: Path,
    start: Optional[int],
    end: Optional[int],
) -> ak.highlevel.Array:
    with uproot.open(fn) as rfile:
        roottree = rfile[conf.loader.rootprefix]
        if start is end is None:
            return roottree.arrays(
                list(conf.loader.braches.values()),
                library="ak",
            )
        elif isinstance(start, int) and isinstance(end, int):
            return roottree.arrays(
                list(conf.loader.braches.values()),
                entry_start=start,
                entry_stop=end,
                library="ak",
            )
        else:
            raise ValueError()


def read_chunks(chunks: List[Tuple[Path, int, int]]) -> ak.highlevel.Array:
    chunks_list = []
    for chunk in chunks:
        chunks_list.append(readpath(*chunk))
    return ak.concatenate(chunks_list)


def path_to_len(fn: Path) -> int:
    with uproot.open(fn) as rfile:
        return rfile[conf.loader.rootprefix].num_entries


file_manager = FileManager(path_to_len=path_to_len)


scaler = ScalerBase(
    file_manager.files,
    file_manager.file_len_dict,
    [
        PowerTransformer(method="box-cox"),
        StandardScaler(),
        StandardScaler(),
        MinMaxScaler(feature_range=(-1, 1)),
    ],
    read_chunks,
    hitlist_to_graph,
)
