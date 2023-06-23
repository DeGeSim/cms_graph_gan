from itertools import combinations

from fgsim.config import conf
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.labels import var_to_label
from fgsim.plot.marginals import ftx_marginals
from fgsim.plot.xyscatter import xy_hist

from .ratioplot import ratioplot


def validation_plots(fig_logger: FigLogger, res: dict, best_last_val: str):
    # res["sim_batch"]._small = Batch.from_data_list(
    #     res["sim_batch"][: conf.loader.batch_size]
    # )
    # res["gen_batch"]._small = Batch.from_data_list(
    #     res["gen_batch"][: conf.loader.batch_size]
    # )
    # for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
    #     v1name = conf.loader.x_features[v1]
    #     v2name = conf.loader.x_features[v2]
    #     cmbname = f"{v1name}_vs_{v2name}"
    # if "val" not in best_last_val:
    #     figure = xyscatter_faint(
    #         sim=res["sim_batch"]_small.x[:, [v1, v2]].cpu().numpy(),
    #         gen=res["gen_batch"]_small.x[:, [v1, v2]].cpu().numpy(),
    #         title=(
    #             f"Scatter points ({conf.loader.n_points}) in batch"
    #             f" ({conf.loader.batch_size})"
    #         ),
    #         v1name=v1name,
    #         v2name=v2name,
    #     )
    #     fig_logger(figure, f"xyscatter_batch_{cmbname}.pdf")

    for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
        figure = xy_hist(
            sim=res["sim_batch"].x[:, [v1, v2]].cpu().numpy(),
            gen=res["gen_batch"].x[:, [v1, v2]].cpu().numpy(),
            title=(
                f"2D Histogram for {conf.loader.n_points} points in"
                f" {conf.loader.test_set_size} events"
            ),
            v1name=var_to_label(v1),
            v2name=var_to_label(v2),
        )
        fig_logger(
            figure,
            f"2dhist_{conf.loader.x_features[v1]}_vs_{conf.loader.x_features[v2]}.pdf",
        )

    if conf.dataset_name == "jetnet" and best_last_val != "val/scaled":
        from fgsim.plot.jetfeatures import jet_features

        for title, fig in jet_features(
            res["sim_batch"],
            res["gen_batch"],
        ).items():
            fig_logger(fig, title)

    for title, fig in ftx_marginals(
        res["sim_batch"],
        res["gen_batch"],
    ).items():
        fig_logger(fig, title)

    val_size = conf.loader.validation_set_size
    for icritic, (sim_crit, gen_crit) in enumerate(
        zip(
            res["sim_crit"].reshape(-1, val_size),
            res["gen_crit"].reshape(-1, val_size),
        )
    ):
        fig_logger(
            ratioplot(sim=sim_crit, gen=gen_crit, title=f"Critic #{icritic} Score"),
            f"critic{icritic}.pdf",
        )
