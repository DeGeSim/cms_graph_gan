import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf
from fgsim.io.sel_loader import scaler
from fgsim.plot import binborders_wo_outliers


class Smoother:
    def __init__(self) -> None:
        if not conf.training.smoothing.active:
            return
        # This code scales the smearning standard deviations
        zero_tf = scaler.transfs.transform(
            torch.tensor([[0.0] * conf.loader.n_features])
        )
        self.scaled_stds = (
            scaler.transfs.transform(
                np.array(conf.training.smoothing.stds).reshape(1, -1)
            )
            - zero_tf
        )

    def smooth_features(self, x, step):
        if not conf.training.smoothing.active:
            return x
        # x_inv_transf = scaler.inverse_transform(x)
        # for idx, name in enumerate(conf.loader.x_features):
        #     self.plotdist(name, x_inv_transf[:, idx])

        x = (
            x
            + torch.from_numpy(
                np.random.multivariate_normal(
                    [0.0] * conf.loader.n_features,
                    np.diag(
                        (
                            self.scaled_stds.squeeze()
                            * self.turnoff(step, conf.training.smoothing.decay)
                        )
                        ** 2
                    ),
                    size=x.shape[:-1],
                )
            )
            .to(x.device)
            .float()
        )
        # x_inv_transf = scaler.inverse_transform(x)
        # for idx, (name, std) in enumerate(
        #     zip(conf.loader.x_features, conf.training.smoothing.stds)
        # ):
        #     self.plotdist(name, x_inv_transf[:, idx], std=std, presmooth=False)
        return x

    def turnoff(self, step, decay):
        if decay == 0:
            return 1
        step *= decay
        if step > np.pi:
            return 0
        else:
            return (np.cos(step) + 1) / 2

    def plotdist(self, name, arr, std=None, presmooth=True):
        arr = arr[:10000]
        # dists = (
        #     torch.cdist(arr.reshape(1, -1, 1), arr.reshape(1, -1, 1))
        #     .reshape(-1)
        #     .cpu()
        #     .numpy()
        # )
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.hist(dists, bins=binbourders_wo_outliers(dists, bins=300),
        # histtype="bar")
        # ax1.set_title("cdist histogram")

        # ax2.set_title("marginal histogram")
        fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
        ax2.hist(
            arr,
            bins=binborders_wo_outliers(arr, bins=300),
            histtype="bar",
        )
        ax2.set_ylabel("Count")
        # ax2.set_xlabel(
        #     {
        #         "E": "E [GeV]",
        #         "x": "x [cm]",
        #         "y": "y [cm]",
        #         "layer": "layer [1]",
        #     }[name]
        # )

        fig.suptitle(
            f"Histogram for {name} w/o smoothing"
            if presmooth
            else f"Histogram for {name} w/ std {std}"
        )
        fig.tight_layout()
        fig.savefig(
            f"wd/{name}-dists.png"
            if presmooth
            else f"wd/{name}-{std}-dists_smoothed.png"
        )
        plt.close()


smoother = Smoother()
smooth_features = smoother.smooth_features
