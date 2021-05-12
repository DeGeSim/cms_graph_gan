import os
from pathlib import Path

import h5py as h5
import torchdata

from .config import conf
from .transform import transform
from .utils.logger import logger

# https://gist.github.com/branislav1991/4c143394bdad612883d148e0617bdccd#file-hdf5_dataset-py


class HDF5Dataset(torchdata.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(
        self,
        file_path,
        transform,
    ):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.xname = conf.loader.xname
        self.yname = conf.loader.yname
        self.h5_file = h5.File(file_path)

        # filename -> length
        self.len = len(self.h5_file[self.xname])
        self.xv = self.h5_file[self.xname]
        self.yv = self.h5_file[self.yname]

    def __getitem__(self, index):
        x = self.xv[index]
        y = self.yv[index]
        return (x, y)

    def __len__(self):
        return self.len


import os

os.chdir(os.path.expanduser("~/fgsim"))

p = Path("wd/forward/Ele_FixedAngle")
assert p.is_dir()
files = sorted(p.glob("**/*.h5"))
dscol = [HDF5Dataset(file_path=fn, transform=transform) for fn in files]

dscol = [e.map(transform) for e in dscol]

print("foo")
