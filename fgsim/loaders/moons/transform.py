import numpy as np
import torch
from sklearn.datasets import make_moons
from torch_geometric.data import Data as Data

from fgsim.config import conf


def transform(_: None) -> Data:
    x1, which_moon = make_moons(conf.loader.n_points)  # , noise = 0.01)
    mu = [0, 0]
    covar = [[0.01, 0], [0, 0.01]]
    jitter = np.random.multivariate_normal(mu, covar, conf.loader.n_points)
    pointcloud = x1 + jitter
    pointcloud = torch.tensor(pointcloud).float()
    graph = Data(x=pointcloud)
    return graph
