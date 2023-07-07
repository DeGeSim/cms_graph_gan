from itertools import combinations

from fgsim.config import conf
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.infolut import var_to_bins, var_to_label
from fgsim.plot.marginals import ftx_marginals
from fgsim.plot.xyscatter import xy_hist

from .ratioplot import ratioplot


def eval_plots(fig_logger: FigLogger, res: dict):
    fig_logger.best_last_val.append("unscaled")
    make_1d_plots(res, fig_logger, "x")
    make_2d_plots(res, fig_logger, "x")

    if conf.dataset_name == "calochallange":
        fig_logger.best_last_val[-1] = "scaled"
        make_1d_plots(res, fig_logger, "x_scaled")
        make_2d_plots(res, fig_logger, "x_scaled")

    if conf.dataset_name == "jetnet":
        fig_logger.best_last_val[-1] = "jn"
        make_jetnet_plots(res, fig_logger)

    fig_logger.best_last_val[-1] = "critics"
    make_critics_plots(res, fig_logger)


def make_1d_plots(res: dict, fig_logger: FigLogger, ftxname: str) -> None:
    fig_logger.best_last_val.append("1D")
    if "unscaled" in fig_logger.best_last_val:
        bins = [var_to_bins(e) for e in conf.loader.x_features]
    else:
        bins = None
    for title, fig in ftx_marginals(
        res["sim_batch"], res["gen_batch"], ftxname, bins
    ).items():
        fig_logger(fig, title)
    fig_logger.best_last_val.pop()


def make_2d_plots(res: dict, fig_logger: FigLogger, ftxname: str) -> None:
    if "unscaled" in fig_logger.best_last_val:
        bins = [var_to_bins(e) for e in conf.loader.x_features]
    else:
        bins = [None for _ in conf.loader.x_features]
    fig_logger.best_last_val.append("2D")
    for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
        figure = xy_hist(
            sim=res["sim_batch"][ftxname][:, [v1, v2]].cpu().numpy(),
            gen=res["gen_batch"][ftxname][:, [v1, v2]].cpu().numpy(),
            title=(
                f"2D Histogram for {conf.loader.n_points} points in"
                f" {conf.loader.test_set_size} events"
            ),
            v1name=var_to_label(v1),
            v2name=var_to_label(v2),
            v1bins=bins[v1],
            v2bins=bins[v2],
        )
        fig_logger(
            figure,
            f"2dhist_{conf.loader.x_features[v1]}_vs_{conf.loader.x_features[v2]}.pdf",
        )
    fig_logger.best_last_val.pop()


def make_critics_plots(res: dict, fig_logger: FigLogger) -> None:
    match conf.command:
        case "train":
            val_size = conf.loader.validation_set_size
        case "test":
            val_size = conf.loader.test_set_size
        case _:
            raise Exception()

    batch_size = conf.loader.batch_size
    fraction = val_size // batch_size
    for icritic, (sim_crit, gen_crit) in enumerate(
        zip(
            res["sim_crit"].reshape(fraction, -1, batch_size).transpose(0, 1),
            res["gen_crit"].reshape(fraction, -1, batch_size).transpose(0, 1),
        )
    ):
        fig_logger(
            ratioplot(
                sim=sim_crit.reshape(-1),
                gen=gen_crit.reshape(-1),
                title=f"Critic #{icritic} Score",
            ),
            f"critic{icritic}.pdf",
        )


def make_jetnet_plots(res: dict, fig_logger: FigLogger) -> None:
    from fgsim.plot.jetfeatures import jet_features

    for title, fig in jet_features(
        res["sim_batch"],
        res["gen_batch"],
    ).items():
        fig_logger(fig, title)
