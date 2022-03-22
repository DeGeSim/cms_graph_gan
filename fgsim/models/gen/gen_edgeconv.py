import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import EdgeConv, knn_graph

from fgsim.config import conf
from fgsim.models.dnn_gen import dnn_gen
from fgsim.monitoring.logger import logger


class ModelClass(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_features: int,
        n_global: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_global = n_global
        self.n_events = conf.loader.batch_size
        self.n_layers = n_layers
        self.z_shape = conf.loader.batch_size, conf.loader.max_points, n_features

        self.output_points = conf.loader.max_points
        logger.debug(f"Generator output will be {self.output_points}")

        # self.dyn_hlvs_layers = nn.ModuleList(
        #     [
        #         DynHLVsLayer(
        #             n_features=n_features,
        #             n_global=n_global,
        #             device=device,
        #             n_events=self.n_events,
        #         )
        #         for _ in range(n_layers)
        #     ]
        # )
        self.pp_convs = nn.ModuleList(
            [
                EdgeConv(
                    nn=dnn_gen(n_features * 2, n_features),  # + n_global
                    aggr="add",
                )
                for _ in range(n_layers)
            ]
        )

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> Batch:

        batch = Batch.from_data_list([Data(x=e) for e in random_vector])

        batch.edge_index = knn_graph(x=batch.x, k=25, batch=batch.batch)

        for ilayer in range(self.n_layers):
            # hlvs = self.dyn_hlvs_layers[ilayer]
            batch.x = self.pp_convs[ilayer](x=batch.x, edge_index=batch.edge_index)

        return batch
