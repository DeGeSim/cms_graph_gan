import multiprocessing

import torch

from ..geo.graph import grid_to_graph
from ..utils.logger import logger


def transform(sample):
    (x, y) = sample
    grap = grid_to_graph(x)
    grap.y = y
    return grap
