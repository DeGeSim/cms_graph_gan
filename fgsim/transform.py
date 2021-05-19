import multiprocessing

import torch

from .geo.graph import grid_to_graph_geo
from .utils.logger import logger

# def transform(sample):
#     (x, y) = sample
#     try:
#         res = (grid_to_graph(x), y)
#     except:
#         logger.warn(f"Error in for y {y} and x \n {x}")
#         exit(1)
#     return res


def transform(sample):
    (x, y) = sample
    grap = grid_to_graph_geo(x)
    grap.y = y
    return grap


# def transform(args):
#     if len(args)>1:
#         sample, batchidx, id = args
#     else:
#         sample = args
#         batchidx = 999
#         id = 999
#     (x, y) = sample
#     if "17" in multiprocessing.current_process().name:
#         logger.warn(f"Problem for number {id} idx {batchidx} for y {y} and x \n {x}.")
#     logger.warn(
#         f"Starting for  number {id} idx {batchidx} name {multiprocessing.current_process().name}:"
#     )
#     try:
#         res = (grid_to_graph(x), torch.tensor(y, dtype=torch.float32))
#     except:
#         logger.warn(f"Error in {id} for y {y} and x \n {x}")
#         exit(1)
#     logger.warn(f"Done for {id}.")
#     return res
