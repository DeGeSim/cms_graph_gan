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
        load_data,
        Xname,
        yname,
        data_cache_size=3,
        transform=None,
    ):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.Xname = Xname
        self.yname = yname
        self.chunksize = conf.model["batch_size"]

        # Search for all h5 files
        p = Path(file_path)
        assert p.is_dir()
        if recursive:
            self.files = sorted(p.glob("**/*.h5"))
        else:
            self.files = sorted(p.glob("*.h5"))
        if len(self.files) < 1:
            raise RuntimeError("No hdf5 datasets found")

        for h5dataset_fp in self.files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
        # filename -> length
        dslenD = {}
        # [posinfilelist]-> startindex
        self.fnidexstart = [0]
        for i, fn in enumerate(self.files):
            with h5.File(fn) as h5_file:
                dslenD[fn] = len(h5_file["ECAL"])
                self.fnidexstart.append(self.fnidexstart[-1] + dslenD[fn])

        self.len = sum([dslenD[e] for e in dslenD])

    # globalindexeventidx-> filename
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

        caloimg = self.get_data(self.Xname, filenameidx)[idx_in_file]
        x = self.transform(caloimg)
        # x = [self.transform(iimg) for iimg in caloimgs]

        # get label
        y = self.get_data(self.yname, filenameidx)[idx_in_file]
        y = torch.tensor(y, dtype=torch.float32)  # .float()
        return (x, y)

    def __len__(self):
        return self.len

    def _add_data_infos(self, file_path, load_data):
        with h5.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            # print(h5_file.items())
            # print(h5_file.keys())

            # for gname, group in h5_file.items():
            #     for dname, ds in group.items():
            for dname, ds in [
                e for e in h5_file.items() if e[0] in (self.Xname, self.yname)
            ]:
                # print(f"ds {dname}:\n {ds}.")
                # print(f"ds type {type(ds)}.")
                # print(f"ds dir {dir(ds)}.")
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds.value, file_path)

                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append(
                    {
                        "file_path": file_path,
                        "type": dname,
                        "shape": ds.shape,
                        "cache_idx": idx,
                    }
                )

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5.File(file_path) as h5_file:
            # for gname, group in h5_file.items():
            #     for dname, ds in group.items():
            for dname, ds in [
                e for e in h5_file.items() if e[0] in (self.Xname, self.yname)
            ]:
                # add data to the data cache and retrieve
                # the cache index
                idx = self._add_to_cache(ds[:], file_path)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(
                    i
                    for i, v in enumerate(self.data_info)
                    if v["file_path"] == file_path
                )

                # the data info should have the same index
                # since we loaded it in the same way
                self.data_info[file_idx + idx]["cache_idx"] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {
                    "file_path": di["file_path"],
                    "type": di["type"],
                    "shape": di["shape"],
                    "cache_idx": -1,
                }
                if di["file_path"] == removal_keys[0]
                else di
                for di in self.data_info
            ]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data."""
        data_info_type = [di for di in self.data_info if di["type"] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
        dataset. This will make sure that the data is loaded in case it is
        not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]["file_path"]
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]["cache_idx"]
        return self.data_cache[fp][cache_idx]


dataset = HDF5Dataset(
    file_path="wd/forward/Ele_FixedAngle",
    recursive=False,
    load_data=False,
    transform=grid_to_graph,
    Xname="ECAL",
    yname="energy",
    data_cache_size=100,
)
