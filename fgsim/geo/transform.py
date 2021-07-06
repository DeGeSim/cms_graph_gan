import torch

from ..geo.graph import grid_to_graph


def transform(sample):
    (x, y) = sample
    grap = grid_to_graph(x)
    grap.y = torch.tensor(y)
    return grap
