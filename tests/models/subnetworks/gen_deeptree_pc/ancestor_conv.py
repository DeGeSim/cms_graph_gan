import torch
from torch_geometric.data import Data

from fgsim.models.subnetworks.gen_deeptree_pc.ancestor_conv import AncestorConv

device = torch.device("cpu")


class IdentityLayer(torch.nn.Module):
    """
    The IdentityLayer is a module that does nothing but
    return the input as the output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.f = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def test_ancestorconv_single_event():
    do_nothing = IdentityLayer()
    ancestor_conv = AncestorConv(do_nothing, do_nothing, add_self_loops=False)

    # Create a graph with 1 event, 2 levels, 2 branches
    # n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    n_events = 1
    graph = Data(
        x=torch.tensor(
            [[1], [2], [5]], dtype=torch.float, device=device, requires_grad=True
        ),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long, device=device),
        edge_attr=torch.tensor([1, 1], dtype=torch.long, device=device),
        event=torch.tensor([0, 0, 0], dtype=torch.long, device=device),
    )
    global_features = torch.tensor(
        [[0.3, 0.2]], dtype=torch.float, device=device
    ).reshape(n_events, n_global)

    res = ancestor_conv(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        event=graph.event,
        global_features=global_features,
    )
    m1 = torch.hstack([graph.x[0], global_features[0], torch.tensor(1)])
    messages = torch.stack([torch.zeros_like(m1), m1, m1])

    res_expect = torch.hstack([graph.x, global_features[graph.event], messages])
    assert torch.all(res == res_expect)


def test_ancestorconv_double_event():
    do_nothing = IdentityLayer()
    ancestor_conv = AncestorConv(do_nothing, do_nothing, add_self_loops=False)

    # Create a graph with 1 event, 2 levels, 2 branches
    # n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    n_events = 2
    graph = Data(
        x=torch.tensor(
            [[1], [1.5], [2], [2.5], [5], [5.5]],
            dtype=torch.float,
            device=device,
            requires_grad=True,
        ),
        edge_index=torch.tensor(
            [[0, 1, 0, 1], [2, 3, 4, 5]], dtype=torch.long, device=device
        ),
        edge_attr=torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device),
        event=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long, device=device),
    )
    global_features = torch.tensor(
        [[0.3, 0.3], [0.1, 0.1]], dtype=torch.float, device=device
    ).reshape(n_events, n_global)

    res = ancestor_conv(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        event=graph.event,
        global_features=global_features,
    )
    m0 = torch.hstack([graph.x[0], global_features[0], torch.tensor(1)])
    m1 = torch.hstack([graph.x[1], global_features[1], torch.tensor(1)])
    messages = torch.stack(
        [
            torch.zeros_like(m0),  # for Node 0
            torch.zeros_like(m1),  # for Node 1
            m0,  # for Node 2
            m1,  # for Node 3
            m0,  # for Node 4
            m1,  # for Node 5
        ]
    )

    res_expect = torch.hstack([graph.x, global_features[graph.event], messages])
    assert torch.all(res == res_expect)


def test_ancestorconv_three_levels():
    do_nothing = IdentityLayer()
    ancestor_conv = AncestorConv(do_nothing, do_nothing, add_self_loops=False)

    # Create a graph with 1 event, 2 levels, 2 branches
    # n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    n_events = 1
    graph = Data(
        x=torch.arange(
            7, dtype=torch.float, device=device, requires_grad=True
        ).reshape(-1, 1),
        edge_index=torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 2, 2], [1, 2, 3, 4, 5, 6, 3, 4, 5, 6]],
            dtype=torch.long,
            device=device,
        ),
        edge_attr=torch.ones(10, dtype=torch.long, device=device),
        event=torch.zeros(7, dtype=torch.long, device=device),
    )
    global_features = torch.tensor(
        [[0.3, 0.2]], dtype=torch.float, device=device
    ).reshape(n_events, n_global)

    res = ancestor_conv(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        event=graph.event,
        global_features=global_features,
    )
    m0 = torch.hstack([graph.x[0], global_features[0], torch.tensor(1)])
    m1 = torch.hstack([graph.x[1], global_features[0], torch.tensor(1)])
    m2 = torch.hstack([graph.x[2], global_features[0], torch.tensor(1)])
    messages = torch.stack(
        [
            torch.zeros_like(m0),  # for Node 0
            m0,  # for Node 1
            m0,  # for Node 2
            m0 + m1,  # for Node 3
            m0 + m1,  # for Node 4
            m0 + m2,  # for Node 5
            m0 + m2,  # for Node 6
        ]
    )

    res_expect = torch.hstack([graph.x, global_features[graph.event], messages])
    assert torch.all(res == res_expect)
