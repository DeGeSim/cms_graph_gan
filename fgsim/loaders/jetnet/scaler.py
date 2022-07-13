from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

from fgsim.config import conf

from .files import files, len_dict


class Scaler:
    def __init__(self) -> None:
        self.scalerpath = Path(conf.path.dataset_processed) / "scaler.gz"
        if not self.scalerpath.is_file():
            if conf.command != "preprocess":
                raise FileNotFoundError()
            self.transfs = None
        else:

            self.transfs = joblib.load(self.scalerpath)

    def save_scaler(
        self,
    ):
        from .seq import read_chunk, transform_wo_scaling

        assert len_dict[files[0]] >= conf.loader.scaling_fit_size
        chk = read_chunk([(Path(files[0]), 0, conf.loader.scaling_fit_size)])
        event_list = [transform_wo_scaling(e) for e in chk]

        # The features need to be converted to numpy immediatly
        # otherwise the queuflow afterwards doesnt work
        pcs = np.vstack([e.x.clone().numpy() for e in event_list])
        if hasattr(event_list[0], "mask"):
            mask = np.hstack([e.mask.clone().numpy() for e in event_list])
            pcs = pcs[mask]

        self.plot_scaling(pcs)
        self.transfs = [
            StandardScaler(),
            StandardScaler(),
            PowerTransformer(method="box-cox", standardize=True),
        ]
        for arr, transf in zip(pcs.T, self.transfs):
            transf.fit(arr.reshape(-1, 1))
        self.plot_scaling(pcs, True)

        joblib.dump(self.transfs, self.scalerpath)

    def transform(self, pcs: np.ndarray):
        assert len(pcs.shape) == 2
        assert pcs.shape[1] == len(self.transfs)
        return np.hstack(
            [
                transf.transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )

    def inverse_transform(self, pcs: np.ndarray):
        assert len(pcs.shape) == 2
        assert pcs.shape[1] == len(self.transfs)
        return np.hstack(
            [
                transf.inverse_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )

    def plot_scaling(self, pcs, post=False):
        if post:
            arr = self.transform(pcs).T
        else:
            arr = pcs.T
        for k, v in zip(conf.loader.cell_prop_keys, arr):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(v, bins=500)
            fig.savefig(
                f"/home/mscham/fgsim/wd/{k}_post.png"
                if post
                else f"/home/mscham/fgsim/wd/{k}_pre.png"
            )
            plt.close(fig)


scaler = Scaler()
