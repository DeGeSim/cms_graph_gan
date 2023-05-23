import torch
from scipy.stats import wasserstein_distance
from torch_geometric.data import Batch
from torch_geometric.nn.pool import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from fgsim.config import conf
from fgsim.models.pool.std_pool import global_std_pool


def scaled_w1(s: torch.Tensor, g: torch.Tensor) -> float:
    assert s.dim() == g.dim() == 1
    s = s.detach().cpu()
    g = g.detach().cpu()
    mean, std = s.mean(), s.std()
    std[std == 0] = 1
    sp = (s - mean) / (std + 1e-4)
    gp = (g - mean) / (std + 1e-4)
    return float(wasserstein_distance(sp.numpy(), gp.numpy()))


def cone_ratio_from_batch(batch: Batch) -> torch.Tensor:
    batchidx = batch.batch
    Ehit = batch.x[:, conf.loader.x_ftx_energy_pos]
    Esum = global_add_pool(Ehit, batchidx)
    x, y = batch.xyz[:, 0], batch.xyz[:, 1]

    # get the center, weighted by energy
    x_center = global_add_pool(x * Ehit, batchidx) / Esum
    y_center = global_add_pool(y * Ehit, batchidx) / Esum

    # hit distance to center
    delta = torch.sqrt(
        (x - x_center[batchidx]) ** 2 + (y - y_center[batchidx]) ** 2
    )
    # energy fraction inside circle around center
    e_small = global_add_pool(Ehit * (delta < 0.2).float(), batchidx) / Esum
    e_large = global_add_pool(Ehit * (delta < 0.3).float(), batchidx) / Esum

    return e_small / e_large


def w1cr(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[float]:
    return scaled_w1(
        cone_ratio_from_batch(sim_batch).reshape(-1),
        cone_ratio_from_batch(gen_batch).reshape(-1),
    )


def fpc_from_batch(batch: Batch) -> torch.Tensor:
    """Get the first principal component from a PC"""
    batchidx = batch.batch.detach().cpu()
    xyz = batch.xyz.detach().cpu()
    means = global_mean_pool(xyz, batchidx)
    stds = global_std_pool(xyz, batchidx)
    deltas = (xyz - means[batchidx]) / stds[batchidx]
    cov = torch.stack(
        [
            torch.stack(
                [
                    global_mean_pool(deltas[:, i] * deltas[:, j], batchidx)
                    # global_add_pool(deltas[:, i] * deltas[:, j], batchidx)
                    # / (batch.ptr[1:]-batch.ptr[:-1] - 1)  # normalize
                    for i in range(3)
                ]
            )
            for j in range(3)
        ]
    ).transpose(0, -1)

    _, e_vec = torch.linalg.eigh(cov)

    # largest_ev = e_val.argmax(-1).reshape(-1, 1, 1)
    # first_pc = e_vec.take_along_dim(largest_ev, -1)
    # (first_pc==e_vec[:,:,[2]]).all() why?? are they sorted?
    # https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html
    # e_vec are sorted, get last colomn
    first_pc = e_vec[:, :, -1]

    # TEST
    # untrfs = deltas[batchidx==0]
    # first pc in the the first component:
    # fct = (first_pc[0].reshape(1, 3) @ untrfs.T).reshape(-1, 1) * first_pc[
    #     0
    # ].reshape(1, 3)
    # assert ((untrfs-fct).std(0)<(untrfs).std(0)).all()
    return first_pc


def w1fpc(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[float]:
    sim_fpc = fpc_from_batch(sim_batch)
    gen_fpc = fpc_from_batch(gen_batch)
    w1s = [scaled_w1(s, g) for s, g in zip(sim_fpc.T, gen_fpc.T)]
    return dict(zip(["x", "y", "z"], w1s))


def cdfdist(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[float]:
    # Eralphz
    cdf_fake = gen_batch.x.cumsum(-1)
    cdf_fake /= cdf_fake[-1].clone()
    cdf_real = sim_batch.x.cumsum(-1)
    cdf_real /= cdf_real[-1].clone()
    cdfdist = (cdf_fake - cdf_real).abs().mean(0)
    return dict(zip(conf.loader.x_features, cdfdist.detach().cpu().numpy()))


def psr_from_batch(batch: Batch) -> dict[str, torch.Tensor]:
    """Get the ratio between half-turnon and
    half turnoff to peak center from a PC"""
    batchidx = batch.batch
    z = batch.xyz[..., -1]
    global_max_pool(z, batchidx)

    zvals = z.unique()
    if len(zvals) > 100:
        raise RuntimeError("To many layers")

    # Construct the matrices to compare which zvalue belongs to which layer
    zval_mat = zvals.reshape(1, -1).repeat(len(z), 1)
    z_mat = z.reshape(-1, 1).repeat(1, len(zvals))

    # compute the histogramm with dim (events,layers)
    hist = global_add_pool((zval_mat == z_mat).float(), batchidx)
    del zval_mat, z_mat
    assert (hist.sum(1) == batch.ptr[1:] - batch.ptr[:-1]).all()

    # # Produce the zshape
    zshape = hist.sum(0)

    # # Produce the zshape centered by the peak
    peak_hits, peak_layer = hist.max(1)
    # produce an index for each layer in each event
    centered_layers_perevent = (
        torch.arange(len(zvals))
        .to(z.device)
        .reshape(1, -1)
        .repeat(len(peak_layer), 1)
    )
    # shift peak to number of layers
    centered_layers_perevent += -peak_layer.reshape(-1, 1) + len(zvals)

    # aggegate for each of the shifted layers
    zshape_centered = global_add_pool(
        hist.reshape(-1), centered_layers_perevent.reshape(-1)
    )

    # Plot to check
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(zshape_centered.cpu(), marker="o", linestyle="none")
    # ax.axvline(len(zvals))
    # ax.axhline(zshape.max().cpu() / 2)
    # fig.savefig("wd/zshape_centered.png")

    # fig, ax = plt.subplots()
    # ax.plot(zshape.cpu())
    # fig.savefig("wd/zshape.png")

    # # Estimate the turnon/turnoff width
    # get a boolean matrix to find the layers with enough hits
    occupied_layers = hist > peak_hits.reshape(-1, 1).repeat(1, len(zvals)) / 2
    eventidx, occ_layer = occupied_layers.nonzero().T
    # find the min and max layer from the bool matrix ath the same time
    minmax = global_max_pool(torch.stack([occ_layer, -occ_layer]).T, eventidx)
    # reverse the -1 trick
    turnoff_layer, turnon_layer = (minmax * torch.tensor([1, -1]).to(z.device)).T
    # distance to the peak
    turnoff_dist = turnoff_layer - peak_layer
    turnon_dist = peak_layer - turnon_layer
    psr = (turnoff_dist + 1) / (turnon_dist + 1)
    # print((turnoff_dist > turnon_dist).float().mean())

    return {
        "peak_layer": peak_layer.float(),
        "psr": psr,
        "turnon_layer": turnon_layer.float(),
        "shape": zshape,
        "cshape": zshape_centered,
    }


def w1z(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    sim_psr = psr_from_batch(sim_batch)
    gen_psr = psr_from_batch(gen_batch)

    w1s = {k: scaled_w1(sim_psr[k], gen_psr[k]) for k in sim_psr}
    return w1s


def w1mar(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    w1s = {
        k: scaled_w1(s, g)
        for s, g, k in zip(sim_batch.x.T, gen_batch.x.T, conf.loader.x_features)
    }
    return w1s
