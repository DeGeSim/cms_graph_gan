import os

import awkward as ak
import numpy as np
import torchdata

from .config import conf
from .transform import transform
from .utils.logger import logger

# https://gist.github.com/branislav1991/4c143394bdad612883d148e0617bdccd#file-hdf5_dataset-py


class sparseds(torchdata.Dataset):
    def __init__(
        self,
        file_path,
    ):
        super().__init__()

        self.file_path = file_path
        self.transform = transform
        self.xname = conf.loader.xname
        self.yname = conf.loader.yname

        logger.info(f"Loading sparse dataset {self.file_path}")
        if not os.path.isfile(self.file_path):
            logger.warn(
                "Sparse Dataset does not exist, starting creation. This will take a while."
            )
            from .write_sparse_ds import write_sparse_ds

            write_sparse_ds()

        self.ds = ak.from_parquet(self.file_path)

        # filename -> length
        self.len = len(self.ds)

    def __getitem__(self, index):
        if type(index) == np.ndarray:
            index=index.tolist()
        return self.ds[index]

    def __len__(self):
        return self.len


dataset = sparseds(file_path=f"wd/{conf.tag}/dssparse.parquet")
