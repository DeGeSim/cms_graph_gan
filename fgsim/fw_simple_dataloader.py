from pathlib import Path

import h5py as h5
import torch

from .config import conf
from .geo.graph import grid_to_graph


# https://gist.github.com/branislav1991/4c143394bdad612883d148e0617bdccd#file-hdf5_dataset-py
class HDF5Dataset(torch.utils.data.Dataset):
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
        recursive,
        Xname,
        yname,
        transform,
    ):
        super().__init__()
        self.transform = transform
        self.Xname = Xname
        self.yname = yname

        # Search for all h5 files
        p = Path(file_path)
        assert p.is_dir()
        if recursive:
            self.files = sorted(p.glob("**/*.h5"))
        else:
            self.files = sorted(p.glob("*.h5"))
        if len(self.files) < 1:
            raise RuntimeError("No hdf5 datasets found")

        # filename -> length
        dslenD = {}
        # [posinfilelist]-> startindex
        self.fnidexstart = [0]
        for i, fn in enumerate(self.files):
            with h5.File(fn) as h5_file:
                dslenD[fn] = len(h5_file["ECAL"])
                self.fnidexstart.append(self.fnidexstart[-1] + dslenD[fn])

        self.len = sum([dslenD[e] for e in dslenD])

    # globalindexeventidx -> filename
    def fn_of_idx(self, idx):
        for i in range(len(self.fnidexstart)):
            # at the end, return the last index
            if i == len(self.fnidexstart) - 1:
                return i
            # check if the starting index of the next file is greater then the given value
            # and return it if true
            if self.fnidexstart[i + 1] > idx:
                return i

    def index_infile(self, idx, filenameidx):
        return idx - self.fnidexstart[filenameidx]

    def __getitem__(self, index):
        # get data
        filenameidx = self.fn_of_idx(index)
        idx_in_file = self.index_infile(index, filenameidx)

        with h5.File(self.files[filenameidx]) as h5_file:
            caloimg = h5_file[self.Xname][idx_in_file]
            y = h5_file[self.yname][idx_in_file]

        x = self.transform(caloimg)
        y = torch.tensor(y, dtype=torch.float32)  # .float()
        return (x, y)

    def __len__(self):
        return self.len


dataset = HDF5Dataset(
    file_path="wd/forward/Ele_FixedAngle",
    recursive=False,
    transform=grid_to_graph,
    Xname="ECAL",
    yname="energy",
)
foo = dataset[0]
