import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from jetnet.datasets import JetNet
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fgsim.config import conf

warnings.filterwarnings("ignore", ".*TypedStorage is deprecated.*")

transfs = [
    StandardScaler(),
    StandardScaler(),
    PowerTransformer(method="box-cox", standardize=True),
]


def contruct_graph_from_row(arg) -> Data:
    x, y = arg
    x, y = x.clone(), y.clone()
    res = Data(
        x=x[x[..., 3].bool(), :3].reshape(-1, 3),
        y=y.reshape(1, -1),
    )
    return res.clone()


class PyGDS(InMemoryDataset):
    def __init__(self):
        root = conf.path.dataset
        self.scaler = PyGScaler(root, transfs)
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        ds_path = Path(conf.path.dataset).expanduser()
        assert ds_path.is_dir()
        files = sorted(ds_path.glob(conf.loader.dataset_glob))
        if len(files) < 1:
            raise RuntimeError("No datasets found")
        return [f for f in files]

    @property
    def processed_file_names(self):
        return "pygds.pt"

    def process(self):
        # Read data into huge `Data` list.
        particle_data, jet_data = JetNet.getData(
            jet_type=conf.loader.jettype,
            data_dir=Path(conf.loader.dataset_path).expanduser(),
            num_particles=conf.loader.n_points,
        )

        data_list = [
            contruct_graph_from_row(e)
            for e in tqdm(
                zip(torch.Tensor(particle_data), torch.Tensor(jet_data)),
                total=len(particle_data),
            )
        ]

        self.data, self.slices = self.collate(data_list)

        train_size = (
            len(data_list)
            - conf.loader.validation_set_size
            - conf.loader.test_set_size
        )
        n_samples = min(train_size, conf.loader.scaling_fit_size)

        self.scaler.fit(self[:n_samples].x)

        self.x = self.scaler.transform(self.x)

        torch.save((self.data, self.slices), self.processed_paths[0])


class PyGScaler:
    def __init__(
        self,
        root,
        transfs,
    ) -> None:
        self.root = root
        self.transfs = transfs
        self.scalerpath = Path(self.root) / "scaler.gz"
        if self.scalerpath.is_file():
            self.transfs = joblib.load(self.scalerpath)

    def fit(self, pcs: torch.Tensor):
        # The features need to be converted to numpy immediatly
        # otherwise the queuflow afterwards doesnt work
        pcs = pcs.numpy()

        self.plot_scaling(pcs)

        assert pcs.shape[1] == conf.loader.n_features
        pcs = np.hstack(
            [
                transf.fit_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        self.plot_scaling(pcs, True)
        joblib.dump(self.transfs, self.scalerpath)

    def transform(self, pcs: torch.Tensor):
        assert len(pcs.shape) == 2
        assert pcs.shape[1] == conf.loader.n_features
        return torch.Tensor(
            np.hstack(
                [
                    transf.transform(arr.reshape(-1, 1))
                    for arr, transf in zip(pcs.numpy().T, self.transfs)
                ]
            )
        )

    def inverse_transform(self, pcs: torch.Tensor):
        assert pcs.shape[-1] == conf.loader.n_features
        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.to("cpu").detach().reshape(-1, conf.loader.n_features).numpy()

        t_stacked = np.hstack(
            [
                transf.inverse_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        return torch.Tensor(t_stacked.reshape(*orgshape)).to(dev)

    def plot_scaling(self, pcs, post=False):
        for k, v in zip(conf.loader.x_features, pcs.T):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(v, bins=500)
            fig.savefig(
                Path(self.root) / f"{k}_post.png"
                if post
                else Path(self.root) / f"{k}_pre.png"
            )
            plt.close(fig)


class PyGLoader:
    def __init__(self) -> None:
        ds = PyGDS()
        self.use_train_for_val = True
        if self.use_train_for_val:
            test_size = conf.loader.test_set_size
            train_size = len(ds) - test_size

            self.ds_train = ds[:train_size]
            self.ds_test = ds[train_size:]
        else:
            val_size = conf.loader.validation_set_size
            test_size = conf.loader.test_set_size
            train_size = len(ds) - val_size - test_size

            self.ds_train = ds[:train_size]
            self.ds_val = ds[train_size : train_size + val_size]
            self.ds_test = ds[train_size + val_size :]

        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=conf.loader.batch_size,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=100,
            num_workers=10,
            drop_last=True,
        )

    def __iter__(self):
        return iter(self.dl_train)

    @property
    def qfseq(self):
        return self

    @property
    def started(self):
        return True

    @property
    def n_grad_steps_per_epoch(self):
        return len(self.dl_train)

    def stop(self):
        del self.dl_train

    def queue_epoch(self, *args, **kwargs):
        pass

    @property
    def validation_batches(self):
        if self.use_train_for_val:
            self.ds_train.shuffle()
            return DataLoader(
                self.ds_train[: conf.loader.validation_set_size],
                batch_size=conf.loader.batch_size,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=100,
                num_workers=5,
            )

        else:
            return DataLoader(
                self.ds_val,
                batch_size=conf.loader.batch_size,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=100,
                num_workers=5,
            )
