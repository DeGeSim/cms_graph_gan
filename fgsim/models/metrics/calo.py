import numpy as np
import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.plot import binborders_wo_outliers, var_to_bins

from .distutils import (
    calc_cdf_dist,
    calc_hist_dist,
    calc_scaled1d_dist,
    calc_wcdf_dist,
)


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

    assert real.shape[-1] == fake.shape[-1]
    assert real.shape[0] >= fake.shape[0]
    # if necessairy, sample r
    if real.shape[0] > fake.shape[0]:
        idx = torch.randperm(real.shape[0])[: fake.shape[0]]
        real = real[idx]

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
    return run_dists(
        sim_batch,
        gen_batch,
        k=None,
        bins=[torch.tensor(var_to_bins(i)).float() for i in range(4)],
    )


def marginalEw(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    # Eralphz
    res_d = {}

    real = sim_batch.x[..., 1:]
    fake = gen_batch.x[..., 1:]
    rw = sim_batch.x[..., 0]
    fw = gen_batch.x[..., 0]

    assert real.shape[-1] == fake.shape[-1]
    assert real.shape[0] >= fake.shape[0]
    # if necessairy, sample r
    if real.shape[0] > fake.shape[0]:
        idx = torch.randperm(real.shape[0])[: fake.shape[0]]
        real = real[idx]
        rw = rw[idx]

    kwargs = {
        "r": real,
        "f": fake,
        "bins": [torch.tensor(var_to_bins(i)).float() for i in range(1, 4)],
        "rw": rw,
        "fw": fw,
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


def sphereratio(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="sphereratio")


def response(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="response")


def fpc(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="fpc")


def showershape(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="showershape")


def nhits(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, np.float32]:
    return run_dists(sim_batch, gen_batch, k="nhits")
