import torch
from torch_geometric.data import Batch
from torch_geometric.nn import knn_graph


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

        nndelta_sim = self.nndist(sim_batch)
        nndelta_gen = self.nndist(gen_batch)

        loss = torch.tensor(0.0).to(gen_batch.x.device)

        for iftx in range(gen_batch.x.shape[-1]):
            nnd_sim_scaled, nnd_gen_scaled = scale_b_to_a(
                nndelta_sim[:, iftx], nndelta_gen[:, iftx]
            )
            loss += wasserstein_distance(nnd_sim_scaled, nnd_gen_scaled)

        return loss

    def nndist(self, batch):
        x = batch.x
        batchidx = batch.batch
        ei = knn_graph(x.clone(), k=3, batch=batchidx, loop=False)
        delta = x[ei[0]] - x[ei[1]]
        return delta.abs()


def cdf(arr):
    # arr = arr.clone() / arr.clone().sum()
    val, _ = arr.sort()
    cdf = val.cumsum(-1)
    cdf /= cdf[-1].clone()
    return cdf


def w1dist(a, b):
    ca = cdf(a)
    cb = cdf(b)
    dist = (ca - cb).abs().mean()
    return dist


def scale_b_to_a(a, b):
    assert not a.requires_grad
    mean, std = a.mean(), a.std()
    assert (std > 1e-6).all()
    sa = (a - mean) / (std + 1e-4)
    sb = (b - mean) / (std + 1e-4)
    return sa, sb


def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)


def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
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
            (torch.tensor([0.0]), u_weights[u_sorter].cumsum())
        )
        u_cdf = u_sorted_cumweights[u_cdf_indices].float() / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_values.numel()
    else:
        v_sorted_cumweights = torch.cat(
            (torch.tensor([0.0]), v_weights[v_sorter].cumsum())
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
