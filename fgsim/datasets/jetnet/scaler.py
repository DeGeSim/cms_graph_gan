from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    StandardScaler,
)

from fgsim.io import ScalerBase

from .graph_transform import events_to_batch
from .readin import file_manager, read_chunks


def Identity(x):
    return x


def DummyTransformer():
    return FunctionTransformer(Identity, Identity)


scaler = ScalerBase(
    files=file_manager.files,
    len_dict=file_manager.file_len_dict,
    transfs_x=[
        StandardScaler(),
        StandardScaler(),
        PowerTransformer(method="box-cox", standardize=True),
    ],
    transfs_y=[
        DummyTransformer(),  # type
        DummyTransformer(),  # pt
        DummyTransformer(),  # eta
        DummyTransformer(),  # mass
        DummyTransformer(),  # num_particles
    ],
    read_chunk=read_chunks,
    events_to_batch=events_to_batch,
)
