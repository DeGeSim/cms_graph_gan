from itertools import combinations

from fgsim.config import conf
from fgsim.plot import FigLogger, hist1d, hist2d, var_to_bins, var_to_label

from .ratioplot import ratioplot


def eval_plots(fig_logger: FigLogger, res: dict):
    fig_logger.prefixes.append("unscaled")
    make_1d_plots(res, fig_logger, "x")
    make_1d_plots(res, fig_logger, "x", True)
    make_2d_plots(res, fig_logger, "x")
    make_2d_plots(res, fig_logger, "x", True)
    fig_logger.prefixes.pop()

    # if conf.dataset_name == "calochallange":
    #     fig_logger.prefixes.append("scaled")
    #     make_1d_plots(res, fig_logger, "x_scaled")
    #     make_2d_plots(res, fig_logger, "x_scaled")
    #     fig_logger.prefixes.pop()

    make_high_level_plots(res, fig_logger)

    if conf.dataset_name == "jetnet":
        fig_logger.prefixes[-1] = "jn"
        make_jetnet_plots(res, fig_logger)

    make_critics_plots(res, fig_logger)


def make_1d_plots(
    res: dict, fig_logger: FigLogger, ftxname: str, energy_weighted=False
) -> None:
    fig_logger.prefixes.append("1D")
    if "unscaled" in fig_logger.prefixes:
        bins = [var_to_bins(e) for e in conf.loader.x_features]
    else:
        bins = None
    for title, fig in hist1d(
        res["sim_batch"], res["gen_batch"], ftxname, bins, energy_weighted
    ).items():
        fig_logger(fig, title)
    fig_logger.prefixes.pop()


def make_high_level_plots(res: dict, fig_logger: FigLogger) -> None:
    fig_logger.prefixes.append("HLV")
    metrics = [e for e in conf.training.val.metrics if "marginal" not in e]
    metric_dict = {}
    for mname in metrics:
        simobj = res["sim_batch"][mname]
        if isinstance(simobj, dict):
            for smname in simobj.keys():
                metric_dict[f"{mname}_{smname}"] = (
                    res["sim_batch"][mname][smname],
                    res["gen_batch"][mname][smname],
                )
        else:
            metric_dict[mname] = (res["sim_batch"][mname], res["gen_batch"][mname])

    for ftn, (sim_arr, gen_arr) in metric_dict.items():
        fig = ratioplot(
            sim=sim_arr.cpu().numpy(),
            gen=gen_arr.cpu().numpy(),
            title=var_to_label(ftn),
        )
        fig_logger(fig, f"hlv_{ftn}.pdf")

    fig_logger.prefixes.pop()


def make_2d_plots(
    res: dict, fig_logger: FigLogger, ftxname: str, energy_weighted=False
) -> None:
    fig_logger.prefixes.append("2D")
    epos = conf.loader.x_ftx_energy_pos
    ename = conf.loader.x_features[epos]

    ftxidxs, ftxnames = zip(
        *list(
            (idx, e)
            for idx, e in enumerate(conf.loader.x_features)
            if not energy_weighted or e != ename
        )
    )

    if "unscaled" in fig_logger.prefixes:
        bins = [var_to_bins(e) for e in conf.loader.x_features]
    else:
        bins = [None for _ in conf.loader.x_features]

    fext = "_Ew" if energy_weighted else ""
    title = (
        f"2D Histogram for {conf.loader.n_points} points in"
        f" {conf.loader.test_set_size} events"
    )
    if energy_weighted:
        title = "Weighted " + title

    for v1, v2 in combinations(ftxidxs, 2):
        if energy_weighted:
            simw = res["sim_batch"][ftxname][:, epos].cpu().numpy()
            genw = res["gen_batch"][ftxname][:, epos].cpu().numpy()
        else:
            simw = None
            genw = None
        figure = hist2d(
            sim=res["sim_batch"][ftxname][:, [v1, v2]].cpu().numpy(),
            gen=res["gen_batch"][ftxname][:, [v1, v2]].cpu().numpy(),
            title=title,
            v1name=var_to_label(v1),
            v2name=var_to_label(v2),
            v1bins=bins[v1],
            v2bins=bins[v2],
            simw=simw,
            genw=genw,
        )
        fig_logger(
            figure,
            f"2dhist_{conf.loader.x_features[v1]}_vs_{conf.loader.x_features[v2]}{fext}.pdf",
        )
    fig_logger.prefixes.pop()


def make_critics_plots(res: dict, fig_logger: FigLogger) -> None:
    fig_logger.prefixes.append("critics")
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
        fig = ratioplot(
            sim=sim_crit.reshape(-1),
            gen=gen_crit.reshape(-1),
            title=f"Critic #{icritic} Score",
        )
        fig_logger(
            fig,
            f"critic{icritic}.pdf",
        )
    fig_logger.prefixes.pop()


def make_jetnet_plots(res: dict, fig_logger: FigLogger) -> None:
    fig_logger.prefixes.append("jetnet")
    from fgsim.plot.jetfeatures import jet_features

    for title, fig in jet_features(
        res["sim_batch"],
        res["gen_batch"],
    ).items():
        fig_logger(fig, title)
    fig_logger.prefixes.pop()
