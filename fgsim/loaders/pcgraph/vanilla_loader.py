import os.path as osp

import torch
import uproot
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from fgsim.config import conf

from .files import files
from .seq import transform


class YourDataset(Dataset):
    def __init__(self):
        super().__init__(conf.loader.dataset_path, transform=transform)

        # test_size = conf.loader.test_set_size
        # validation_size = conf.loader.validation_set_size
        # self.validation = self.fullds[:validation_size]
        # self.test = self.fullds[validation_size : (validation_size + test_size)]
        # self.train = self.fullds[(validation_size + test_size) :]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with uproot.open(raw_path) as rfile:
                roottree = rfile[conf.loader.rootprefix]
                data = roottree.arrays(
                    list(conf.loader.braches.values()),
                    library="ak",
                )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    @property
    def raw_file_names(self):
        return [files[0], files[1]]

    @property
    def processed_file_names(self):
        return ["pp1.pt", "pp2.pt", "pp3.pt"]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data


your_dataset = YourDataset()
print("processing")
your_dataset.process()
your_data_loader = DataLoader(
    your_dataset,
    batch_size=conf.loader.batch_size,
    shuffle=False,
    num_workers=30,
    prefetch_factor=10,
)
