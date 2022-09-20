import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from fgsim.config import conf, device


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()

        # Shape of the random vector
        self.z_shape = conf.loader.batch_size, 1, 1
        self.par = torch.nn.Parameter(torch.tensor([0.5]))
        self.__cached_batch = Batch.from_data_list(
            [
                Data(
                    x=torch.zeros(
                        (conf.loader.n_points, conf.loader.n_features),
                        dtype=torch.float,
                        requires_grad=True,
                        device=device,
                    )
                )
                for _ in range(conf.loader.batch_size)
            ]
        )

    def forward(self, *args, **kwargs) -> Batch:
        return self.__cached_batch.clone()
