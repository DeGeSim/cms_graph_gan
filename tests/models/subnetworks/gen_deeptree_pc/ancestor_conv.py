import pytest
import torch
from torch_geometric.data import Data

from fgsim.models.common.deeptree import DeepConv

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


@pytest.fixture
def default_pars():
    return {
        "n_features": 1,
        "n_branches": 2,
        "n_levels": 2,
        "n_global": 2,
        "n_cond": 1,
        "batch_size": 1,
    }


@pytest.fixture
def ancestor_conv(default_pars):
    do_nothing = IdentityLayer()
    ancestor_conv = DeepConv(
        in_features=default_pars["n_features"],
        out_features=default_pars["n_features"],
        n_global=default_pars["n_global"],
        n_cond=default_pars["n_cond"],
        nns="both",
        add_self_loops=False,
        msg_nn_include_edge_attr=True,
        msg_nn_include_global=True,
        upd_nn_include_global=True,
        residual=False,
    )
    ancestor_conv.msg_nn = do_nothing
    ancestor_conv.update_nn = do_nothing
    return ancestor_conv


@pytest.fixture
def graph(default_pars):
    # Create a graph with 1 event, 2 levels, 2 branches
    return Data(
        tftx=torch.tensor([[1], [2], [5]], dtype=torch.float, requires_grad=True),
        cond=torch.zeros(
            (default_pars["batch_size"], default_pars["n_cond"]),
            dtype=torch.float,
            requires_grad=True,
        ),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_attr=torch.tensor([[1], [1]], dtype=torch.long),
        tbatch=torch.tensor([0, 0, 0], dtype=torch.long),
    ).to(device)


def test_ancestorconv_single_event(ancestor_conv, graph):
    # n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    batch_size = 1

    global_features = torch.tensor(
        [[0.3, 0.2]], dtype=torch.float, device=device
    ).reshape(batch_size, n_global)

    graph.cond = torch.tensor([[5]], dtype=torch.float, device=device).reshape(
        batch_size, 1
    )

    res = ancestor_conv(
        x=graph.tftx,
        cond=graph.cond,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.tbatch,
        global_features=global_features,
    )
    m1 = torch.hstack([graph.tftx[0], global_features[0], torch.tensor(1)])
    messages = torch.stack([torch.zeros_like(m1), m1, m1])

    res_expect = torch.hstack(
        [
            graph.tftx,
            graph.cond[graph.tbatch],
            global_features[graph.tbatch],
            messages,
        ]
    )

    assert torch.allclose(res, res_expect)


def test_ancestorconv_double_event(ancestor_conv):
    # Create a graph with 1 event, 2 levels, 2 branches
    # n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    batch_size = 2

    graph = Data(
        tftx=torch.tensor(
            [[1], [1.5], [2], [2.5], [5], [5.5]],
            dtype=torch.float,
            device=device,
            requires_grad=True,
        ),
        cond=torch.tensor(
            [[5] * batch_size], dtype=torch.float, device=device
        ).reshape(batch_size, 1),
        edge_index=torch.tensor(
            [[0, 1, 0, 1], [2, 3, 4, 5]], dtype=torch.long, device=device
        ),
        edge_attr=torch.tensor(
            [[1], [1], [1], [1]], dtype=torch.long, device=device
        ),
        tbatch=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long, device=device),
    )
    global_features = torch.tensor(
        [[0.3, 0.3], [0.1, 0.1]], dtype=torch.float, device=device
    ).reshape(batch_size, n_global)

    res = ancestor_conv(
        x=graph.tftx,
        cond=graph.cond,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.tbatch,
        global_features=global_features,
    )
    m0 = torch.hstack([graph.tftx[0], global_features[0], torch.tensor(1)])
    m1 = torch.hstack([graph.tftx[1], global_features[1], torch.tensor(1)])
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

    res_expect = torch.hstack(
        [
            graph.tftx,
            graph.cond[graph.tbatch],
            global_features[graph.tbatch],
            messages,
        ]
    )
    assert torch.all(res == res_expect)


def test_ancestorconv_three_levels(ancestor_conv):
    # Create a graph with 1 event, 2 levels, 2 branches
    # n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    batch_size = 1

    graph = Data(
        tftx=torch.arange(
            7, dtype=torch.float, device=device, requires_grad=True
        ).reshape(-1, 1),
        cond=torch.tensor(
            [[5] * batch_size], dtype=torch.float, device=device
        ).reshape(batch_size, 1),
        edge_index=torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 2, 2], [1, 2, 3, 4, 5, 6, 3, 4, 5, 6]],
            dtype=torch.long,
            device=device,
        ),
        edge_attr=torch.ones(10, dtype=torch.long, device=device).reshape(-1, 1),
        tbatch=torch.zeros(7, dtype=torch.long, device=device),
    )
    global_features = torch.tensor(
        [[0.3, 0.2]], dtype=torch.float, device=device
    ).reshape(batch_size, n_global)

    res = ancestor_conv(
        x=graph.tftx,
        cond=graph.cond,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.tbatch,
        global_features=global_features,
    )
    m0 = torch.hstack([graph.tftx[0], global_features[0], torch.tensor(1)])
    m1 = torch.hstack([graph.tftx[1], global_features[0], torch.tensor(1)])
    m2 = torch.hstack([graph.tftx[2], global_features[0], torch.tensor(1)])
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

    res_expect = torch.hstack(
        [
            graph.tftx,
            graph.cond[graph.tbatch],
            global_features[graph.tbatch],
            messages,
        ]
    )
    assert torch.all(res == res_expect)


def test_ancestorconv_all_modes():
    # Create a graph with 1 event, 2 levels, 2 branches
    n_features = 1
    # n_branches = 2
    # n_levels = 2
    n_global = 2
    n_cond = 1
    batch_size = 1

    graph = Data(
        tftx=torch.arange(
            7, dtype=torch.float, device=device, requires_grad=True
        ).reshape(-1, 1),
        cond=torch.tensor(
            [[5] * batch_size], dtype=torch.float, device=device
        ).reshape(batch_size, 1),
        edge_index=torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 2, 2], [1, 2, 3, 4, 5, 6, 3, 4, 5, 6]],
            dtype=torch.long,
            device=device,
        ),
        edge_attr=torch.ones(10, dtype=torch.long, device=device).reshape(-1, 1),
        tbatch=torch.zeros(7, dtype=torch.long, device=device),
    )
    global_features = torch.tensor(
        [[0.3, 0.2]], dtype=torch.float, device=device
    ).reshape(batch_size, n_global)

    for add_self_loops in [True, False]:
        for msg_nn_bool in [True, False]:
            for upd_nn_bool in [True, False]:
                for msg_nn_include_edge_attr in [True, False]:
                    for msg_nn_include_global in [True, False]:
                        for upd_nn_include_global in [True, False]:
                            for residual in [True, False]:
                                if not upd_nn_bool and upd_nn_include_global:
                                    continue
                                if not msg_nn_bool:
                                    if (
                                        msg_nn_include_edge_attr
                                        or msg_nn_include_global
                                    ):
                                        continue
                                if msg_nn_bool and upd_nn_bool:
                                    nns = "both"
                                elif msg_nn_bool:
                                    nns = "msg"
                                elif upd_nn_bool:
                                    nns = "upd"
                                else:
                                    continue
                                ancestor_conv = DeepConv(
                                    in_features=n_features,
                                    out_features=n_features,
                                    n_global=n_global,
                                    n_cond=n_cond,
                                    add_self_loops=add_self_loops,
                                    nns=nns,
                                    msg_nn_include_edge_attr=msg_nn_include_edge_attr,
                                    msg_nn_include_global=msg_nn_include_global,
                                    upd_nn_include_global=upd_nn_include_global,
                                    residual=residual,
                                )
                                kwargs = {
                                    "x": graph.tftx,
                                    "cond": graph.cond,
                                    "edge_index": graph.edge_index,
                                    "batch": graph.tbatch,
                                }
                                kwargs["edge_attr"] = graph.edge_attr
                                kwargs["global_features"] = global_features
                                ancestor_conv(**kwargs)


# def test_ancester_conv_by_training():
#     from torch_geometric.data import Batch, Data
#     from torch_geometric.nn.conv import GINConv

#     from fgsim.models.branching.branching import BranchingLayer, Tree
#     from fgsim.models.dnn_gen import dnn_gen
#     from fgsim.models.layer.ancestor_conv import AncestorConv

#     n_features = 2
#     batch_size = 1
#     n_branches = 2
#     n_levels = 3
#     n_global = 0
#     device = torch.device("cpu")
#     tree = Tree(
#         batch_size=batch_size,
#         n_features=n_features,
#         n_branches=n_branches,
#         n_levels=n_levels,
#         device=device,
#     )
#     anc_conv = AncestorConv(
#         in_features=n_features,
#         out_features=n_features,
#         n_global=n_global,
#         msg_nn_include_global=False,
#         upd_nn_include_global=False,
#         msg_nn_include_edge_attr=False,
#     ).to(device)

#     branching_layer = BranchingLayer(
#         tree=tree,
#         proj_nn=dnn_gen(
#             n_features + n_global,
#             n_features * n_branches,
#             n_layers=4,
#         ).to(device),
#     )
#     tbatch = Batch.from_data_list([Data(tftx=torch.tensor([[1.0, 1.0]]))])
#     global_features = torch.tensor([[]])
#     target = torch.tensor([[4.0, 7.0], [5.0, 1.0], [2.0, 2.5], [3.0, 3.5]])
#     loss_fn = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(anc_conv.parameters())
#     for _ in range(10000):
#         optimizer.zero_grad()
#         b1 = branching_layer(tbatch, global_features)
#         b2 = branching_layer(b1, global_features)
#         res = anc_conv(
#             tftx=b2.tftx,
#             edge_index=b2.edge_index,
#             tbatch=b2.event,
#             global_features=global_features,
#         )
#         loss = loss_fn(res[3:, :], target)
#         loss.backward()
#         optimizer.step()
#         if _ % 1000 == 0:
#             print(loss)
#         if torch.allclose(target, res[3:, :], rtol=1e-4, atol=1e-4):
#             return
#     raise Exception
