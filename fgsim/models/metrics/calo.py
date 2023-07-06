import numpy as np
import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.plot.xyscatter import binborders_wo_outliers

from .distutils import (
    calc_cdf_dist,
    calc_hist_dist,
    calc_scaled1d_dist,
    calc_wcdf_dist,
)

mar_bins = [
    torch.linspace(0, 6000, 100).float(),  # E
    torch.linspace(0, 44, 45).float(),  # z
    torch.linspace(0, 15, 16).float(),  # alpha
    torch.linspace(0, 8, 9).float(),  # r
]


def run_dists(sim_batch, gen_batch, k, bins=None):
    if k is not None:
        if isinstance(sim_batch[k], dict):
            ftxnames = sim_batch[k].keys()
        else:
            ftxnames = []
    else:
        ftxnames = conf.loader.x_features

    if bins is None:
        bins = []
        if len(ftxnames) == 0:
            arr = sim_batch[k].cpu().numpy()
            bins.append(torch.tensor(binborders_wo_outliers(arr, bins=300)).float())
        else:
            for fn in ftxnames:
                arr = sim_batch[k][fn].cpu().numpy()
                bins.append(
                    torch.tensor(binborders_wo_outliers(arr, bins=300)).float()
                )
    if k is None:
        real = sim_batch.x
        fake = gen_batch.x
    else:
        if len(ftxnames) == 0:
            real = sim_batch[k].unsqueeze(-1)
            fake = gen_batch[k].unsqueeze(-1)
        else:
            real = torch.stack([sim_batch[k][ftxn] for ftxn in ftxnames], -1)
            fake = torch.stack([gen_batch[k][ftxn] for ftxn in ftxnames], -1)

    dists_d = {
        distname: fct(r=real, f=fake, bins=bins)
        for distname, fct in zip(
            ["cdf", "sw1", "histd"],
            [calc_cdf_dist, calc_scaled1d_dist, calc_hist_dist],
        )
    }
    res_d = {}
    for dname, darr in dists_d.items():
        if len(ftxnames) == 0:
            res_d[f"{dname}"] = darr[0]
            continue
        for iftx, ftxname in enumerate(ftxnames):
            res_d[f"{ftxname}/{dname}"] = darr[iftx]

    return res_d


def marginal(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k=None, bins=mar_bins)


def marginalEw(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    # Eralphz
    res_d = {}
    kwargs = {
        "r": sim_batch.x[..., 1:],
        "f": gen_batch.x[..., 1:],
        "bins": mar_bins[1:],
        "rw": sim_batch.x[..., 0],
        "fw": gen_batch.x[..., 0],
    }
    cdfdist = calc_wcdf_dist(**kwargs)
    sw1dist = calc_scaled1d_dist(**kwargs)
    histdist = calc_hist_dist(**kwargs)
    for (
        k,
        vcdf,
        vsw1d,
        vhistd,
    ) in zip(conf.loader.x_features, cdfdist, sw1dist, histdist):
        res_d[f"{k}/cdf"] = vcdf
        res_d[f"{k}/sw1"] = vsw1d
        res_d[f"{k}/histd"] = vhistd

    return res_d


# global variables
# Marginal Variables


def coneratio(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="coneratio")


def response(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="response")


def fpc(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="fpc")


def showershape(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="showershape")
