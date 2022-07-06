from pathlib import Path

import joblib
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from torch.multiprocessing import Pool

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
        from .seq import aggregate_to_batch, read_chunk, transform_wo_scaling

        assert len_dict[files[0]] >= conf.loader.scaling_fit_size
        chk = read_chunk([(Path(files[0]), 0, conf.loader.scaling_fit_size)])
        with Pool(15) as p:
            event_list = p.map(transform_wo_scaling, chk)
        batch = aggregate_to_batch(event_list)
        pcs = batch.x.numpy()
        # E, x, y, z = pcs.T
        # import matplotlib.pyplot as plt
        # for k,v in zip(["E","x","y","z"],comb_transf.transform(pcs).T):
        #     fig, ax = plt.subplots(figsize =(10, 7))
        #     ax.hist(v,bins=100)
        # fig.savefig(f"/home/mscham/fgsim/wd/{k}_post.png")

        E_transf = PowerTransformer(method="box-cox")
        x_transf = StandardScaler()
        y_transf = StandardScaler()
        z_transf = MinMaxScaler()
        self.comb_transf = make_column_transformer(
            (E_transf, [0]),
            (x_transf, [1]),
            (y_transf, [2]),
            (z_transf, [3]),
        )
        self.comb_transf.fit(pcs)
        joblib.dump(self.comb_transf, self.scalerpath)

    def transform(self, *args, **kwargs):
        return self.comb_transf.transform(*args, **kwargs)


scaler = Scaler()
