import awkward as ak
import torch
import uproot
from torch.utils.data import DataLoader, Dataset

from fgsim.config import conf
from fgsim.loaders.pcgraph.files import files

from .seq import transform

# from torch_geometric.data import Dataset
# from torch_geometric.loader import DataLoader


class YourDataset(Dataset):
    def __init__(self):
        arr_list = []
        for file_path in [files[0]]:
            with uproot.open(file_path) as rfile:
                roottree = rfile[conf.loader.rootprefix]
                arr_list.append(
                    roottree.arrays(
                        list(conf.loader.braches.values()),
                        library="ak",
                    )
                )
        self.fullds = ak.concatenate(arr_list)
        test_size = conf.loader.test_set_size
        validation_size = conf.loader.validation_set_size
        self.validation = self.fullds[:validation_size]
        self.test = self.fullds[validation_size : (validation_size + test_size)]
        self.train = self.fullds[(validation_size + test_size) :]
        self.__len = len(self.train)

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.train[idx]
        x = transform(x)

        return x


your_dataset = YourDataset()
your_data_loader = DataLoader(
    your_dataset,
    batch_size=conf.loader.batch_size,
    shuffle=False,
    num_workers=35,
    prefetch_factor=259,
)
