import torch
from torch_geometric.nn import global_add_pool, global_mean_pool

from fgsim.config import conf
from fgsim.models.dnn_gen import dnn_gen


class ModelClass(torch.nn.Module):
    def __init__(self, activation):
        super(ModelClass, self).__init__()
        self.activation = getattr(torch.nn, activation)()
        n_features = conf.loader.n_features
        self.hlv_dnn = dnn_gen(
            n_features * 2, 1, n_layers=4, activation_last_layer=self.activation
        )

    def forward(self, batch):
        x = batch.x
        x = torch.hstack(
            [
                global_add_pool(x, batch.batch, size=batch.num_graphs),
                global_mean_pool(x, batch.batch, size=batch.num_graphs),
            ]
        )
        x = self.hlv_dnn(x)
        return x
