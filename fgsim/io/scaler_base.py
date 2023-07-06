from pathlib import Path
from typing import Callable, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf


class ScalerBase:
    def __init__(
        self,
        files: List[Path],
        len_dict: Dict,
        read_chunk: Callable,
        transform_wo_scaling: Callable,
        transfs_x,
        transfs_y,
    ) -> None:
        self.files = files
        self.len_dict = len_dict
        self.transfs_x = transfs_x
        self.transfs_y = transfs_y
        self.read_chunk = read_chunk
        self.transform_wo_scaling = transform_wo_scaling
        self.scalerpath = Path(conf.path.dataset_processed) / "scaler.gz"

        assert len(self.transfs_x) == conf.loader.x_features
        assert len(self.transfs_y) == conf.loader.y_features

        if conf.command != "preprocess" and conf.loader.preprocess_training:
            if not self.scalerpath.is_file():
                raise FileNotFoundError(f"No scalar found at {self.scalerpath}")
            else:
                self.transfs_x, self.transfs_y = joblib.load(self.scalerpath)

    def fit(self, saveplots=False):
        assert self.len_dict[self.files[0]] >= conf.loader.scaling_fit_size
        chk = self.read_chunk(
            [(Path(self.files[0]), 0, conf.loader.scaling_fit_size)]
        )
        event_list = [self.transform_wo_scaling(e) for e in chk]

        # The features need to be converted to numpy immediatly
        # otherwise the queuflow afterwards doesnt work
        for x_or_y in ["x", "y"]:
            pcs = np.vstack([e[x_or_y].clone().numpy() for e in event_list]).astype(
                "float64"
            )
            if hasattr(event_list[0], "mask"):
                mask = np.hstack([e.mask.clone().numpy() for e in event_list])
                pcs = pcs[mask]

            if saveplots:
                self.plot_scaling(pcs, x_or_y)
            assert pcs.shape[1] == len(conf.loader[x_or_y + "_features"])
            transfs = self.transfs_x if x_or_y == "x" else self.transfs_y
            res = np.hstack(
                [
                    transf.fit_transform(arr.reshape(-1, 1))
                    for arr, transf in zip(pcs.T, transfs)
                ]
            )
            if not np.isfinite(res).all():
                raise RuntimeError("Result not finite")
            if saveplots:
                self.plot_scaling(res, x_or_y, True)

    def save_scaler(self):
        self.fit(True)
        joblib.dump((self.transfs_x, self.transfs_y), self.scalerpath)

    def transform(self, pcs: torch.Tensor, x_or_y: str):
        assert pcs.dim() == 2
        n_features = len(conf.loader[x_or_y + "_features"])
        assert pcs.shape[1] == n_features
        dev = pcs.device
        pcs = pcs.cpu().numpy().astype("float64")

        transfs = self.transfs_x if x_or_y == "x" else self.transfs_y
        res = np.hstack(
            [
                transf.transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, transfs)
            ]
        )
        if not np.isfinite(res).all():
            raise RuntimeError("Result not finite")
        return torch.Tensor(res).float().to(dev)

    def inverse_transform(self, pcs: torch.Tensor, x_or_y: str):
        assert pcs.dim() == 2
        n_features = len(conf.loader[x_or_y + "_features"])
        assert pcs.shape[1] == n_features
        dev = pcs.device
        pcs = pcs.cpu().numpy().astype("float64")

        transfs = self.transfs_x if x_or_y == "x" else self.transfs_y
        res = np.hstack(
            [
                transf.inverse_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, transfs)
            ]
        )
        if not np.isfinite(res).all():
            raise RuntimeError("Result not finite")
        return torch.from_numpy(res).float().to(dev)

    def plot_scaling(self, pcs, x_or_y: str, post: bool = False):
        if x_or_y == "x":
            feature_names = conf.loader.x_features
        else:
            feature_names = conf.loader.y_features
        for k, v in zip(feature_names, pcs.T):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(v, bins=500)
            fn = Path(conf.path.dataset_processed)
            if post:
                fn = fn / f"{x_or_y}_{k}_post.pdf"
            else:
                fn = fn / f"{x_or_y}_{k}_pre.pdf"
            fig.savefig(str(fn))
            plt.close(fig)
