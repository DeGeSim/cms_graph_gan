import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, knn_graph

from fgsim.config import conf
from fgsim.io.sel_loader import scaler

eidx = conf.loader.x_ftx_energy_pos


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_batch: Batch,
        sim_batch: Batch,
        **kwargs,
    ):
        assert gen_batch.x.requires_grad

        dev = sim_batch.x.device
        posidx = [i for i in range(gen_batch.x.shape[-1]) if i != eidx]

        loss = torch.tensor(0.0).to(gen_batch.x.device)
        for fridx in [[eidx], posidx]:
            nndelta_sim = self.nndist(sim_batch, fridx)
            nndelta_gen = self.nndist(gen_batch, fridx)
            nnd_sim_scaled, nnd_gen_scaled = scale_b_to_a(nndelta_sim, nndelta_gen)
            if fridx == [eidx]:
                loss += wmse(nnd_sim_scaled, nnd_gen_scaled)
            else:
                sw = self.inv_scale_hitE(sim_batch).to(dev)
                gw = self.inv_scale_hitE(gen_batch).to(dev)

                assert (sw > 0).all() and (gw > 0).all()
                loss += wmse(nnd_sim_scaled, nnd_gen_scaled, sw, gw)

        return loss

    def inv_scale_hitE(self, batch):
        return torch.tensor(
            scaler.transfs_x[eidx].inverse_transform(
                batch.x[:, [eidx]].detach().cpu().numpy()
            )
        ).squeeze()

    def nndist(self, batch, slice):
        x = batch.x[:, slice]
        batchidx = batch.batch
        ei = knn_graph(x.clone(), k=3, batch=batchidx, loop=False)
        delta = (x[ei[0]] - x[ei[1]]).abs().mean(1)
        delta_aggr = global_add_pool(delta, ei[1])
        return delta_aggr


# def cdf(arr):
#     # arr = arr.clone() / arr.clone().sum()
#     val, _ = arr.sort()
#     cdf = val.cumsum(-1)
#     cdf /= cdf[-1].clone()
#     return cdf


# def w1dist(a, b):
#     ca = cdf(a)
#     cb = cdf(b)
#     dist = (ca - cb).abs().mean()
#     return dist


def scale_b_to_a(a, b):
    assert not a.requires_grad
    mean, std = a.mean(), a.std()
    assert (std > 1e-6).all()
    sa = (a - mean) / (std + 1e-4)
    sb = (b - mean) / (std + 1e-4)
    return sa, sb


def wmse(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(2, u_values, v_values, u_weights, v_weights)


def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)


def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    dev = u_values.device
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)

    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values)

    # Get the respective positions of the values of u and v
    #  among the values of both distributions.
    u_cdf_indices = torch.searchsorted(
        u_values[u_sorter], all_values[:-1], right=True
    )
    v_cdf_indices = torch.searchsorted(
        v_values[v_sorter], all_values[:-1], right=True
    )

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices.float() / u_values.numel()
    else:
        u_sorted_cumweights = torch.cat(
            (torch.tensor([0.0]).to(dev), u_weights[u_sorter].cumsum(0))
        )
        u_cdf = u_sorted_cumweights[u_cdf_indices].float() / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_values.numel()
    else:
        v_sorted_cumweights = torch.cat(
            (torch.tensor([0.0]).to(dev), v_weights[v_sorter].cumsum(0))
        )
        v_cdf = v_sorted_cumweights[v_cdf_indices].float() / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    if p == 1:
        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas)
    if p == 2:
        return torch.sqrt(torch.sum(torch.square(u_cdf - v_cdf) * deltas))
    return torch.pow(
        torch.sum(torch.pow(torch.abs(u_cdf - v_cdf), p) * deltas), 1 / p
    )
