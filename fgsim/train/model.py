from functools import reduce

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from ..config import conf, device
from ..geo.graph import num_node_features

imgpixels = reduce(lambda a, b: a * b, conf["mapper"]["calo_img_shape"])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, 1)

    def forward(self, data):
        resL = []
        for graphprops in data:
            (
                feature_mtx,
                adj_mtx_coo,
                _,# inner_edges_per_layer,
                _,# forward_edges_per_layer,
                _,# backward_edges_per_layer,
            ) = graphprops
            feature_mtx = feature_mtx.to(device)
            adj_mtx_coo = adj_mtx_coo.to(device)
            feature_mtx = torch.squeeze(feature_mtx)
            adj_mtx_coo = torch.squeeze(adj_mtx_coo)

            graph = Data(
                x=feature_mtx.t().contiguous(), edge_index=adj_mtx_coo.t().contiguous()
            )
            x, edge_index = graph.x, graph.edge_index
            num_nodes, num_node_features = feature_mtx.shape

            # adj_sparse = torch.sparse.FloatTensor(
            #     adj_mtx_coo,
            #     torch.ones((adj_mtx_coo.shape[1]), device=device),
            #     torch.Size((num_nodes, num_nodes)),
            # )
            # adj_sparse = torch.sparse_coo_tensor(
            #     indices=adj_mtx_coo,
            #     values=torch.ones((adj_mtx_coo.shape[1]), device=device),
            #     size=torch.Size((num_nodes, num_nodes)),
            #     # dtype=bool,
            #     requires_grad=False,
            # )
            # feature_mtx = torch.tensor(feature_mtx,requires_grad=True)
            # feature_mtx=feature_mtx.T
            # d=adj_sparse.to_dense()
            x = self.conv1(x.T, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
            resL.append(torch.sum(x))
        return torch.stack(resL)

        # for layer, sel in layerselmap:
        #     # 1 In layer Message pass
        #     inlayerMessagePass(layer)
        #     # 2 forward pass
        #     layerForward(layer)

    # def inlayerMessagePass(layer):
    #     sel = layerselmap[layer]


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr="add")  # "Add" aggregation (Step 5).
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]

#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j


# class Generator(nn.Module):
#     def __init__(self, nz):
#         super(Generator, self).__init__()
#         self.nz = conf["model"]["gan"]["nz"]
#         self.main = nn.Sequential(
#             nn.Linear(self.nz, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 100),
#             nn.LeakyReLU(0.2),
#             nn.Linear(100, imgpixels),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         # change the shape of the output to the shape of the calorimeter image
#         # the first dimension = number of events is inferred by the -1 value
#         return self.main(x).view(-1, *conf["mapper"]["calo_img_shape"]).float()


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.n_input = imgpixels
#         self.main = nn.Sequential(
#             nn.Linear(self.n_input, 100),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(100, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # flatten the image
#         # pass the tensor through the discrimnator
#         return self.main(x.view(-1, imgpixels).float())
