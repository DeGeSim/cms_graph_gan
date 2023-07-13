import torch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool, global_max_pool

from fgsim.config import conf


def analyze_layers(batch: Batch) -> dict[str, torch.Tensor]:
    """Get the ratio between half-turnon and
    half turnoff to peak center from a PC"""
    batchidx = batch.batch
    device = batch.x.device
    z = batch.xyz[..., -1]

    # global_max_pool(z, batchidx)

    zvals = z.unique()
    n_layers = len(zvals)
    if n_layers > 100:
        raise RuntimeError("To many layers")

    # Construct the matrices to compare which zvalue belongs to which layer
    zval_mat = zvals.reshape(1, -1).repeat(len(z), 1)
    z_mat = z.reshape(-1, 1).repeat(1, n_layers)

    # compute the histogramm with dim (events,layers)
    hist = global_add_pool((zval_mat == z_mat).float(), batchidx)
    del zval_mat, z_mat
    assert (hist.sum(1) == batch.ptr[1:] - batch.ptr[:-1]).all()

    max_hits_per_event, peak_layer_per_event = hist.max(1)

    # produce an index for each layer in each event
    # centered_layers_perevent = (
    #     torch.arange(n_layers).to(device).reshape(1, -1).repeat(n_events, 1)
    # )
    # # shift peak to number of layers
    # centered_layers_perevent += -peak_layer_per_event.reshape(-1, 1) + n_layers
    # # separate idx by event
    # eventshift = (
    #     (torch.arange(n_events) * n_layers * 2)
    #     .to(device)
    #     .reshape(-1, 1)
    #     .repeat(1, n_layers)
    # )
    # centered_layers_perevent += eventshift

    # # aggegate for each of the shifted layers
    # zshape_centered = global_add_pool(
    #     hist.reshape(-1), centered_layers_perevent.reshape(-1)
    # )

    # # Estimate the turnon/turnoff width
    # get a boolean matrix to find the layers with enough hits
    occupied_layers = (
        hist > max_hits_per_event.reshape(-1, 1).repeat(1, n_layers) / 2
    )
    eventidx, occ_layer = occupied_layers.nonzero().T
    # find the min and max layer from the bool matrix ath the same time
    minmax = global_max_pool(torch.stack([occ_layer, -occ_layer]).T, eventidx)
    # reverse the -1 trick
    turnoff_layer, turnon_layer = (minmax * torch.tensor([1, -1]).to(device)).T

    # distance to the peak
    turnoff_dist = turnoff_layer - peak_layer_per_event
    turnon_dist = peak_layer_per_event - turnon_layer
    psr = (turnoff_dist + 1) / (turnon_dist + 1)

    return {
        "peak_layer": peak_layer_per_event.float(),
        "psr": psr.float(),
        "turnon_layer": turnon_layer.float(),
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
