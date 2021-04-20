from copy import deepcopy

import h5py as h5
import numba as nb
import numpy as np
import torch

from .geo.fw_loader import graph, num_node_features, num_nodes

# filelist = [
#     f"wd/forward/Ele_FixedAngle/EleEscan_{i}_{j}.h5"
#     # for i in range(1, 9)
#     for i in range(1, 2)
#     # for j in range(1, 11)
#     for j in range(1, 2)
# ]


# def data_generator():
#     for fn in filelist:
#         f = h5.File(fn, "r")
#         caloimgs = f["ECAL"]
#         energies = f["energy"]
#         caloimgs = np.swapaxes(caloimgs, 1, 3)
#         for energy, img in zip(energies, caloimgs):
#             yield (energy, img)


# data_gen = data_generator()


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self):
        "Initialization"
        self.filelist = [
            f"wd/forward/Ele_FixedAngle/EleEscan_{i}_{j}.h5"
            # for i in range(1, 9)
            for i in range(1, 2)
            # for j in range(1, 11)
            for j in range(1, 2)
        ]
        self.entries_per_file = []
        for fn in self.filelist:
            with h5.File(fn, "r") as f:
                self.entries_per_file.append(len(f["energy"]))

        self.list_IDs = list(range(sum(self.entries_per_file)))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        i = 0
        while index > self.entries_per_file[i]:
            i = i + 1
            index = index - self.entries_per_file[i]

        with h5.File(self.filelist[i], "r") as f:
            caloimg = f["ECAL"][i]
            energie = f["energy"][i]

        caloimg = np.swapaxes(caloimg, 0, 2).flatten()

        X = torch.zeros((num_nodes, num_node_features),dtype=torch.float32)
        X[:, 0] = torch.tensor(caloimg, dtype=torch.float32)
        y = torch.tensor(energie,dtype=torch.float32)

        return X, y
