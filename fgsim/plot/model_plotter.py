from itertools import combinations

import torch

from fgsim.plot.binborders import binborders_wo_outliers, chip_to_binborders


class ModelPlotter:
    def __init__(self) -> None:
        self.arrlist = []
        self.n_features = 3  # conf.loader.n_features
        self.combinations = list(combinations(list(range(self.n_features)), 2))
        self.active: bool = False

    def save_tensor(self, name: str, arr: torch.Tensor):
        if not self.active:
            return
        assert len(arr.shape) == 2
        self.arrlist.append(
            [name, arr[..., : self.n_features].detach().cpu().numpy()]
        )

    def plot_model_outputs(self):
        if not self.active:
            raise Exception
        # self.arrlist = self.arrlist[: (len(self.arrlist) // 2)]
        from matplotlib import pyplot as plt

        plt.cla()
        plt.clf()
        n_combs = len(self.combinations)
        n_arrs = len(self.arrlist)
        fig, axs = plt.subplots(n_arrs, n_combs, figsize=(6 * n_combs, 4 * n_arrs))

        for iarr in range(n_arrs):
            arr = self.arrlist[iarr][1]
            arr = arr.reshape(-1, arr.shape[-1])
            title = self.arrlist[iarr][0]
            for icomb in range(n_combs):
                axes = axs[iarr][icomb]
                x = arr[:, self.combinations[icomb][0]]
                y = arr[:, self.combinations[icomb][1]]

                # x_name = conf.loader.x_features[self.combinations[icomb][0]]
                # y_name = conf.loader.x_features[self.combinations[icomb][1]]
                x_name = f"feature # {self.combinations[icomb][0]}"
                y_name = f"feature # {self.combinations[icomb][1]}"

                xedges = binborders_wo_outliers(x)
                yedges = binborders_wo_outliers(y)
                axes.hist2d(
                    chip_to_binborders(x, xedges),
                    chip_to_binborders(y, yedges),
                    bins=[xedges, yedges],
                )
                axes.set_xlabel(x_name)
                axes.set_ylabel(y_name)
                axes.set_title(title)

        self.arrlist = []
        fig.tight_layout()
        return fig


model_plotter = ModelPlotter()
