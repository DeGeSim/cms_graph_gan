import torch
import torch_geometric
from torch import Tensor

# def global_std_pool(x: Tensor, batch: Tensor, size: Optional[int] = None) -> Tensor:
#     dim = -1 if x.dim() == 1 else -2
#     size = int(batch.max().item() + 1) if size is None else size
#     return torch_scatter.scatter_std(x, batch, dim=dim, dim_size=size)


def global_mad_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    means = torch_geometric.nn.global_mean_pool(x, batchidx)
    deltas = (means[batchidx] - x).abs()
    return torch_geometric.nn.global_mean_pool(deltas, batchidx)


def global_var_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    means = torch_geometric.nn.global_mean_pool(x, batchidx)
    deltas = torch.pow(means[batchidx] - x, 2)
    return torch_geometric.nn.global_mean_pool(deltas, batchidx)


def global_std_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    return torch.sqrt(global_var_pool(x, batchidx))


global_width_pool = global_mad_pool
# from torch_scatter import scatter_sum
# import torch

# x = torch.normal(0, 1, (200,), requires_grad=False)
# width = torch.tensor(2.0, requires_grad=True)
# shift = torch.tensor(4.0, requires_grad=True)

# xprime = x * width + shift
# aggr = scatter_std(
#     xprime.reshape(-1, 1),
#     torch.zeros(200, dtype=torch.long),
#     dim=-2,
# ).sum()
# aggr.backward()
# print(width.grad, shift.grad)
