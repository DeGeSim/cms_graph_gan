import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import GINConv

from fgsim.models.subnetworks.gen_deeptree_pc.ancestor_conv import AncestorConvLayer

device = torch.device("cpu")


class IdentityLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def test_AncestorConvLayer_simple():
    do_nothing = IdentityLayer()
    ancestor_conv = AncestorConvLayer(do_nothing, do_nothing)

    # Create a graph with 1 event, 2 levels, 2 branches
    graph = Data(
        x=torch.tensor(
            [[1], [2], [5]], dtype=torch.float, device=device, requires_grad=True
        ),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long, device=device),
        edge_attr=torch.tensor([1, 1], dtype=torch.long, device=device),
        event=torch.tensor([0, 0, 0], dtype=torch.long, device=device),
    )
    global_features = torch.tensor([[]], dtype=torch.float, device=device).reshape(
        3, -1
    )

    res = ancestor_conv(graph.x, graph.edge_index, graph.event, global_features)

    gin_conv = GINConv(do_nothing)
    res_gin = gin_conv(graph.x, graph.edge_index)
    assert torch.all(res == res_gin)


# def test_AncestorConvLayer_2():
#     do_nothing = IdentityLayer()
#     ancestor_conv = AncestorConvLayer(do_nothing, do_nothing)

#     # Create a graph with 1 event, 2 levels, 2 branches
#     graph = Data(
#         x=torch.tensor(
#             [[1], [2], [5]], dtype=torch.float, device=device, requires_grad=True
#         ),
#         edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long, device=device),
#         edge_attr=torch.tensor([1, 1], dtype=torch.long, device=device),
#         event=torch.tensor([0, 0, 0], dtype=torch.long, device=device),
#     )
#     global_features = torch.tensor([[]], dtype=torch.float, device=device)
#     res = ancestor_conv(graph.x, graph.edge_index, graph.event, global_features)

#     gin_conv = GINConv(do_nothing)
#     res_gin = gin_conv(graph.x, graph.edge_index)
#     assert torch.all(res == res_gin)
