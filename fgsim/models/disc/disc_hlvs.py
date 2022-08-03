import torch
from torch_geometric.nn import global_add_pool, global_mean_pool

from fgsim.config import conf
from fgsim.models.common import FFN


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        n_features = conf.loader.n_features
        self.hlv_dnn = FFN(
            n_features * 2, 1, activation_last_layer=torch.nn.Identity()
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
