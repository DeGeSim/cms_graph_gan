import math
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import torch_geometric

from ..config import conf
from ..geo.transform import transform
from ..utils.logger import logger
from ..utils.thread_or_process import pname


class Chunk_Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_path,
        chunk,
        transform,
    ):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.chunk = chunk
        self.start = self.chunk[0]
        self.end = self.chunk[1]
        self.len = self.chunk[1] - self.chunk[0]

    def _loadfile(self):
        logger.debug(
            f"{pname()}: loading {h5.File(self.file_path)} chunk {self.chunk}"
        )
        with h5.File(self.file_path) as h5_file:
            self.x = h5_file[conf.loader.xname][self.chunk[0] : self.chunk[1]]
            self.y = h5_file[conf.loader.yname][self.chunk[0] : self.chunk[1]]

        logger.debug(
            f"{pname()}: done loading {h5.File(self.file_path)} chunk {self.chunk}"
        )

    def __getitem__(self, index):
        if not hasattr(self, "x"):
            self._loadfile()

        # logger.debug(f"{thread_or_process()}: tranforming {index}")
        if callable(self.transform):
            res = self.transform((self.x[index], self.y[index]))
        else:
            res = (self.x[index], self.y[index])
        # logger.debug(f"{thread_or_process()}: done tranforming {index}")
        return res

    def __len__(self):
        return self.len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter((self[i - self.chunk[0]] for i in range(iter_start, iter_end)))


# %%
# Search for all h5 files
p = Path(conf.datasetpath)
assert p.is_dir()
files = sorted(p.glob("**/*.h5"))
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")


chunksize = conf.loader.chunksize
nentries = 10000

chunks = [
    (i * chunksize, (i + 1) * chunksize) for i in range(nentries // chunksize)
] + [(nentries // chunksize * chunksize, nentries)]

# Data Loader copies the dataset once for each worker
# Each chuck is read by each process
# => Loading the file in __get_item__ of Chunk_Dataset does not work because
# each process gets copy of the BufferedShuffleDataset (ChainDataset(Chunk_Dataset))
# and the are first evaluated at runtime
# => Initializing the chunks in a generator passed to ChainDataset does not work
# because the generator is copied between the processes and evaluated separatly


chunk_ds_L = [
    Chunk_Dataset(fn, chunk, transform) for chunk in chunks for fn in files
]

logger.debug("All datasets initialized")

np.random.shuffle(chunk_ds_L)


chained_ds = torch.utils.data.ChainDataset(chunk_ds_L)

# Loader queries batch_size entries at once from the dataset => use buffer_size>> batch_size
buffered_ds = torch.utils.data.BufferedShuffleDataset(
    chained_ds, buffer_size=conf.loader.buffer_size
)


def get_loader():
    loader = torch_geometric.data.DataLoader(
        buffered_ds,
        batch_size=conf.loader.batch_size,
        num_workers=conf.loader.num_workers,
    )
    return loader
    # Dataset is copied for each process => Used memory: buffer_size * num_workers


def dataset_generator():
    collected_e = []
    for ch_ds in chunk_ds_L:
        for e in ch_ds:
            collected_e.append(e)
            if len(collected_e) == conf.loader.batch_size:
                res = torch_geometric.data.Batch().from_data_list(collected_e)
                collected_e = []
                yield res
