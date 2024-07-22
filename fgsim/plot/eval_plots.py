from itertools import combinations
from typing import Union

import numpy as np
import torch

from fgsim.config import conf
from fgsim.plot import (
    FigLogger,
    hist1d,
    hist2d,
    ratioplot,
    var_to_bins,
    var_to_label,
)


def to_np(a: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return a.detach().cpu().numpy()


def eval_plots(fig_logger, res: dict):
    make_1d_plots(res, fig_logger, False)
    make_1d_plots(res, fig_logger, True)
    make_1d_plots(res, fig_logger, False, True)

    make_high_level_plots(res, fig_logger)
    make_critics_plots(res, fig_logger)

    make_2d_plots(res, fig_logger, False)
    make_2d_plots(res, fig_logger, True)
    make_2d_plots(res, fig_logger, False, True)


def make_1d_plots(
    res: dict, fig_logger, scaled: bool, energy_weighted=False
) -> None:
    ftxname = "x_scaled" if scaled else "x"
    fig_logger.prefixes.append("1D_" + ("scaled" if scaled else "unscaled"))
    if not scaled:
        bins = [var_to_bins(e) for e in conf.loader.x_features]
    else:
        bins = [None for _ in conf.loader.x_features]
    for title, fig in hist1d(
        res["sim_batch"],
        res["gen_batch"],
        ftxname,
        bins,
        energy_weighted,
    ).items():
        fig_logger(fig, title)
    fig_logger.prefixes.pop()


def make_high_level_plots(res: dict, fig_logger: FigLogger) -> None:
    fig_logger.prefixes.append("HLV")
    metrics = res["sim_batch"]["hlv"].keys()
    metric_dict = {}
    for mname in metrics:
        sim_obj = res["sim_batch"]["hlv"][mname]
        gen_obj = res["gen_batch"]["hlv"][mname]
        if isinstance(sim_obj, dict):
            for smname in sim_obj.keys():
                metric_dict[f"{mname}_{smname}"] = (
                    sim_obj[smname],
                    gen_obj[smname],
                )
        else:
            metric_dict[mname] = (sim_obj, gen_obj)

    for ftn, (sim_arr, gen_arr) in metric_dict.items():
        fig = ratioplot([to_np(sim_arr), to_np(gen_arr)], ftn=ftn)
        fig_logger(fig, f"hlv_{ftn}.pdf")

    fig_logger.prefixes.pop()


def x_from_batch(batch, ftxname, var, n=100):
    assert batch.batch[-1] > n
    last_point_idx = (batch.batch < n).int().argmin()
    return to_np(batch[ftxname][:last_point_idx, var])


def make_2d_plots(
    res: dict, fig_logger: FigLogger, scaled: bool, energy_weighted=False
) -> None:
    ftxname = "x_scaled" if scaled else "x"
    fig_logger.prefixes.append("2D_" + ("scaled" if scaled else "unscaled"))
    epos = conf.loader.x_ftx_energy_pos
    ename = conf.loader.x_features[epos]

    ftxidxs, ftxnames = zip(
        *list(
            (idx, e)
            for idx, e in enumerate(conf.loader.x_features)
            if not energy_weighted or e != ename
        )
    )

    if not scaled:
        bins = [var_to_bins(e) for e in conf.loader.x_features]
    else:
        bins = [None for _ in conf.loader.x_features]

    fext = "_Ew" if energy_weighted else ""
    title = (
        f"2D Histogram for {conf.loader.n_points} points in"
        f" {conf.loader.test_set_size} events"
    )
    cbtitle = None
    if energy_weighted:
        title = "Weighted " + title
        cbtitle = var_to_label(ename) + " per Shower"

    n = 100

    for v1, v2 in combinations(ftxidxs, 2):
        if energy_weighted:
            simw = x_from_batch(res["sim_batch"], ftxname, epos, n) / n
            genw = x_from_batch(res["gen_batch"], ftxname, epos, n) / n
        else:
            simw = None
            genw = None
        figure = hist2d(
            sim=x_from_batch(res["sim_batch"], ftxname, [v1, v2], n),
            gen=x_from_batch(res["gen_batch"], ftxname, [v1, v2], n),
            v1name=var_to_label(v1),
            v2name=var_to_label(v2),
            v1bins=bins[v1],
            v2bins=bins[v2],
            simw=simw,
            genw=genw,
            cbtitle=cbtitle,
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
            [to_np(sim_crit.reshape(-1)), to_np(gen_crit.reshape(-1))],
            ftn=f"Critic \\#{icritic} Score",
        )
        fig_logger(
            fig,
            f"critic{icritic}.pdf",
        )
    fig_logger.prefixes.pop()
