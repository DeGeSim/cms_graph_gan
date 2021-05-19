import gc
import time
from multiprocessing import Pool
from pathlib import Path

import awkward as ak
import h5py
import numpy as np

from .config import conf
from .geo.graph import grid_to_graph_ak
from .utils.logger import logger
from .utils.memory import memGB, memReport

# https://gist.github.com/branislav1991/4c143394bdad612883d148e0617bdccd#file-hdf5_dataset-py

# Decision to create grid_to_graph_ak instead of grid_to_graph_np:
# np
# fgsim - INFO - Reading wd/forward/Ele_FixedAngle/EleEscan_1_1.h5
# fgsim - INFO - Reading took 24.45
# fgsim - INFO - Mapping took 76.18
# fgsim - INFO - Create ak.Array took 592.15
# fgsim - INFO - Appending took 0.0
# ak
# fgsim - INFO - Reading wd/forward/Ele_FixedAngle/EleEscan_1_1.h5
# fgsim - INFO - Reading took 25.13
# fgsim - INFO - Mapping took 178.58
# fgsim - INFO - Create ak.Array took 5.96
# fgsim - INFO - Appending took 3.67
# if foo == "np":
#     res = p.map(grid_to_graph_np, xv)
#     (
#         feature_mtx,
#         adj_mtx_coo,
#         inner_edges_per_layer,
#         forward_edges_per_layer,
#         backward_edges_per_layer,
#     ) = zip(*res)
#     endmap = time.time()
#     logger.debug(f"Mapping took {np.round(endmap-endread ,2)}")

#     arr = ak.Array(
#         {
#             "feature_mtx": feature_mtx,
#             "adj_mtx_coo": adj_mtx_coo,
#             "inner_edges_per_layer": inner_edges_per_layer,
#             "forward_edges_per_layer": forward_edges_per_layer,
#             "backward_edges_per_layer": backward_edges_per_layer,
#             conf.loader.yname: yv,
#         }
#     )


def write_sparse_ds():
    p = Path("wd/forward/Ele_FixedAngle")
    assert p.is_dir()
    files = sorted(p.glob("**/*.h5"))

    outarr = None
    for ifile, fn in enumerate(files):
        logger.info(f"Reading {fn} ({ifile}/{len(files)})")
        startread = time.time()
        with h5py.File(fn, "r") as inds:
            xv = inds[conf.loader.xname][:]
            yv = inds[conf.loader.yname][:]
        endread = time.time()
        logger.debug(f"Reading took {np.round(endread-startread ,2)}")

        with Pool(30) as p:
            res = p.map(grid_to_graph_ak, xv)
        endmap = time.time()
        logger.debug(f"Mapping took {np.round(endmap-endread ,2)}")
        arr = ak.concatenate(res, axis=0, merge=True)
        arr[conf.loader.yname] = yv

        arrtime = time.time()
        logger.debug(f"Create ak.Array took {np.round(arrtime-endmap ,2)}")
        if outarr is None:
            outarr = arr
        else:
            outarr = ak.concatenate((outarr, arr), axis=0, merge=True)
        appendtime = time.time()
        logger.debug(f"Appending took {np.round(appendtime-arrtime ,2)}")
        gc.collect()
        memReport()
        if memGB() > 60:
            break
    ak.to_parquet(outarr, f"wd/{conf.tag}/dssparse.parquet")
