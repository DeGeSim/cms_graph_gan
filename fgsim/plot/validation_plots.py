from itertools import combinations
from pathlib import Path
from typing import Optional

from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.labels import var_to_label
from fgsim.plot.marginals import ftx_marginals
from fgsim.plot.xyscatter import xy_hist


def validation_plots(
    train_log: TrainLog,
    sim_batch: Batch,
    gen_batch: Batch,
    plot_path: Optional[Path],
    best_last_val: str,
    step: int,
):
    fig_logger = FigLogger(train_log, plot_path, best_last_val, step)

    # sim_batch_small = Batch.from_data_list(sim_batch[: conf.loader.batch_size])
    # gen_batch_small = Batch.from_data_list(gen_batch[: conf.loader.batch_size])
    # for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
    #     v1name = conf.loader.x_features[v1]
    #     v2name = conf.loader.x_features[v2]
    #     cmbname = f"{v1name}_vs_{v2name}"
    # if "val" not in best_last_val:
    #     figure = xyscatter_faint(
    #         sim=sim_batch_small.x[:, [v1, v2]].cpu().numpy(),
    #         gen=gen_batch_small.x[:, [v1, v2]].cpu().numpy(),
    #         title=(
    #             f"Scatter points ({conf.loader.n_points}) in batch"
    #             f" ({conf.loader.batch_size})"
    #         ),
    #         v1name=v1name,
    #         v2name=v2name,
    #         step=step,
    #     )
    #     fig_logger(figure, f"xyscatter_batch_{cmbname}.pdf")

    for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
        figure = xy_hist(
            sim=sim_batch.x[:, [v1, v2]].cpu().numpy(),
            gen=gen_batch.x[:, [v1, v2]].cpu().numpy(),
            title=(
                f"2D Histogram for {conf.loader.n_points} points in"
                f" {conf.testing.n_events} events"
            ),
            v1name=var_to_label(v1),
            v2name=var_to_label(v2),
            step=step,
        )
        fig_logger(
            figure,
            f"2dhist_{conf.loader.x_features[v1]}_vs_{conf.loader.x_features[v2]}.pdf",
        )

    if conf.loader_name == "jetnet" and best_last_val != "val/scaled":
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
