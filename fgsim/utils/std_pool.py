from typing import Optional

from torch import Tensor
from torch_scatter import scatter_std


def global_std_pool(
    x: Tensor, batch: Optional[Tensor], size: Optional[int] = None
) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.MaxAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each element to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.max(dim=dim, keepdim=x.dim() <= 2)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter_std(x, batch, dim=dim, dim_size=size)
