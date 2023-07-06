import torch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool, global_max_pool

from fgsim.config import conf


def analyze_shower(batch: Batch) -> dict[str, torch.Tensor]:
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
    # zshape = hist.sum(0)

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

    # # aggegate for each of the shifted layers
    # zshape_centered = global_add_pool(
    #     hist.reshape(-1), centered_layers_perevent.reshape(-1)
    # )

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
        "psr": psr.float(),
        "turnon_layer": turnon_layer.float(),
        # "shape": zshape, # wrong shape
        # "zshape_centered": zshape_centered, wrong shape
        # TODO
    }


def cone_ratio(batch: Batch) -> torch.Tensor:
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


def response(batch: Batch) -> torch.Tensor:
    batchidx = batch.batch
    Ehit = batch.x[:, conf.loader.x_ftx_energy_pos]
    Esum = global_add_pool(Ehit, batchidx)

    return Esum / batch.y[:, conf.loader.y_features.index("E")]
