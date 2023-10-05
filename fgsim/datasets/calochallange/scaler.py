import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)

from fgsim.config import conf
from fgsim.io.dequantscaler import dequant_stdscale
from fgsim.io.scaler_base import ScalerBase

from .graph_transform import events_to_batch
from .readin import file_manager, read_chunks


def Identity(x):
    return x


def LimitForBoxCox(x):
    return np.clip(x, -19, None)


hitE_tf = make_pipeline(
    PowerTransformer(method="box-cox", standardize=False),
    FunctionTransformer(Identity, LimitForBoxCox, validate=True),
    # SplineTransformer(),
    StandardScaler(),
)

E_tf = make_pipeline(
    PowerTransformer(method="box-cox", standardize=False),
    QuantileTransformer(output_distribution="normal"),
)
num_particles_tf = make_pipeline(
    *dequant_stdscale((0, conf.loader.n_points + 1)),
    QuantileTransformer(output_distribution="normal"),
)
E_per_hit_tf = make_pipeline(
    PowerTransformer(method="box-cox", standardize=False),
    QuantileTransformer(output_distribution="normal"),
)


scaler = ScalerBase(
    files=file_manager.files,
    len_dict=file_manager.file_len_dict,
    transfs_x=[
        hitE_tf,  # h_energy
        make_pipeline(*dequant_stdscale()),  # z
        make_pipeline(*dequant_stdscale()),  # alpha
        make_pipeline(*dequant_stdscale()),  # r
    ],
    transfs_y=[
        E_tf,  # Energy
        num_particles_tf,  # num_particles
        E_per_hit_tf,  # Energy pre hit
    ],
    read_chunk=read_chunks,
    events_to_batch=events_to_batch,
)
