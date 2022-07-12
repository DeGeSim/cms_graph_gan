from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.multiprocessing import Pool
from torch_geometric.data import Batch

from fgsim.config import conf

from .files import files, len_dict


class Scaler:
    def __init__(self) -> None:
        self.scalerpath = Path(conf.path.dataset_processed) / "scaler.gz"
        if not self.scalerpath.is_file():
            if conf.command != "preprocess":
                raise FileNotFoundError()
            self.comb_transf = None
        else:
            self.comb_transf = joblib.load(self.scalerpath)

    def save_scaler(
        self,
    ):
        from .seq import read_chunk, transform_wo_scaling

        assert len_dict[files[0]] >= conf.loader.scaling_fit_size
        chk = read_chunk([(Path(files[0]), 0, conf.loader.scaling_fit_size)])
        with Pool(conf.loader.num_workers_transform) as p:
            event_list = p.map(transform_wo_scaling, chk)

        batch = Batch.from_data_list(event_list)
        pcs = batch.x[batch.mask].numpy()

        self.plot_scaling(pcs)

        self.comb_transf = make_column_transformer(
            (StandardScaler(), [0]),
            (StandardScaler(), [1]),
            (PowerTransformer(method="box-cox", standardize=True), [2]),
        )
        self.comb_transf.fit(pcs)
        self.plot_scaling(pcs, True)

        joblib.dump(self.comb_transf, self.scalerpath)

    def transform(self, *args, **kwargs):
        return self.comb_transf.transform(*args, **kwargs)

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
