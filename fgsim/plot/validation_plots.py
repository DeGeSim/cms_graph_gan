from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.marginals import ftx_marginals
from fgsim.plot.xyscatter import xy_hist, xyscatter_faint


class FigLogger:
    def __init__(self, train_log, plot_path, best_last_val, step) -> None:
        self.train_log = train_log
        self.plot_path = plot_path
        self.best_last_val = best_last_val
        self.step = step

    def __call__(self, figure, filename):
        if self.plot_path is not None:
            figure.savefig((self.plot_path / filename).with_suffix(".png"), dpi=150)
        self.train_log.log_figure(
            figure_name=f"val/{filename}"
            if self.best_last_val == "val"
            else f"test/{self.best_last_val}/{filename}",
            figure=figure,
            overwrite=False,
            step=self.step,
        )
        logger.info(filename)
        plt.close(figure)


def validation_plots(
    train_log: TrainLog,
    sim_batch: Batch,
    gen_batch: Batch,
    plot_path: Optional[Path],
    best_last_val: str,
    step: int,
):
    fig_logger = FigLogger(train_log, plot_path, best_last_val, step)

    sim_batch_small = Batch.from_data_list(sim_batch[: conf.loader.batch_size])
    gen_batch_small = Batch.from_data_list(gen_batch[: conf.loader.batch_size])

    from itertools import combinations

    for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
        v1name = conf.loader.x_features[v1]
        v2name = conf.loader.x_features[v2]
        cmbname = f"{v1name}_vs_{v2name}"

        if "val" not in best_last_val:
            figure = xyscatter_faint(
                sim=sim_batch_small.x[:, [v1, v2]].cpu().numpy(),
                gen=gen_batch_small.x[:, [v1, v2]].cpu().numpy(),
                title=(
                    f"Scatter points ({conf.loader.n_points}) in batch"
                    f" ({conf.loader.batch_size})"
                ),
                v1name=v1name,
                v2name=v2name,
                step=step,
            )
            fig_logger(figure, f"xyscatter_batch_{cmbname}.pdf")

        figure = xy_hist(
            sim=sim_batch.x[:, [v1, v2]].cpu().numpy(),
            gen=gen_batch.x[:, [v1, v2]].cpu().numpy(),
            title=(
                f"2D Histogram for {conf.loader.n_points} points in"
                f" {conf.testing.n_events} events"
            ),
            v1name=v1name,
            v2name=v2name,
            step=step,
        )
        fig_logger(figure, f"{cmbname}.pdf")

    if conf.loader_name == "jetnet":
        from fgsim.plot.jetfeatures import jet_features

        for title, fig in jet_features(
            sim_batch,
            gen_batch,
            step=step,
        ).items():
            fig_logger(fig, title)

    for title, fig in ftx_marginals(
        sim_batch,
        gen_batch,
        step=step,
    ).items():
        fig_logger(fig, title)
