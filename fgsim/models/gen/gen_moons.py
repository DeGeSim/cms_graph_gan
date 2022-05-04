import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch_geometric.data import Batch, Data

from fgsim.config import conf


class ModelClass(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # self.z_shape = (
        #     conf.loader.batch_size,
        #     conf.loader.max_points,
        #     conf.loader.n_features,
        # )
        #
        self.z_shape = (1,)

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> Batch:
        batch_size = conf.loader.batch_size
        max_points = conf.loader.max_points
        mu = [0, 0]

        x, _ = make_moons(max_points * batch_size)
        jitter = np.random.multivariate_normal(
            mu, [[0.01, 0], [0, 0.01]], max_points * batch_size
        )
        pc_moom = (
            torch.tensor(x + jitter).float().reshape(batch_size, max_points, 2)
        )
        moons = Batch.from_data_list([Data(x=e) for e in pc_moom]).to(
            random_vector.device
        )

        return moons
