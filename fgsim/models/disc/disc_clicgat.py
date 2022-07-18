import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GINConv, global_add_pool

from fgsim.config import conf, device
from fgsim.models.ffn import FFN
from fgsim.models.pooling.dyn_hlvs import DynHLVsLayer


class ModelClass(torch.nn.Module):
    def __init__(self, n_features, n_global, n_prop, n_nn):
        super(ModelClass, self).__init__()

        self.n_global = n_global
        self.n_features = n_features
        self.n_prop = n_prop
        self.n_nn = n_nn

        self.conv = GINConv(FFN(n_features + n_global, n_features), train_eps=True)
        self.gat = GATConv(n_features, n_features)
        self.final_dnn = FFN(n_features + n_global, 1)

        self.dynhlvs_dnn = DynHLVsLayer(
            conf.loader.n_features, n_global, conf.loader.batch_size, device
        )

    def forward(self, batch: Batch):
        x, edge_index, batchidxs = batch.x, batch.edge_index, batch.batch

        for _ in range(self.n_prop):
            hlvs = self.dynhlvs_dnn(x, batchidxs)
            x = torch.hstack(
                (
                    x,
                    torch.repeat_interleave(hlvs, batch.ptr[1], dim=0),
                )
            )
            x = self.gat(x, edge_index)
            x = self.conv(x, edge_index)

        hlvs = self.dynhlvs_dnn(x, batchidxs)
        x = global_add_pool(x, batchidxs, size=batch.num_graphs)

        x = self.final_dnn(torch.hstack((x, hlvs)))
        return x
