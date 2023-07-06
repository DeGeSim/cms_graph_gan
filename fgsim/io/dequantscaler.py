from functools import partial

import numpy as np
import scipy
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .idxscale import IdxToScale


def dequant(x):
    noise = np.random.rand(*x.shape)
    return x.astype("float") + noise


def requant(x):
    return np.floor(x)


def forward(x, lower, dist):
    return (x - lower) / dist


def backward(x, lower, dist):
    return x * dist + lower


def dequant_stdscale(inputrange=None) -> list:
    if inputrange is None:
        scaletf = IdxToScale((0, 1))
    else:
        lower, upper = inputrange
        dist = upper - lower

        scaletf = FunctionTransformer(
            partial(forward, lower=lower, dist=dist),
            partial(backward, lower=lower, dist=dist),
            check_inverse=True,
            validate=True,
        )
    tfseq = [
        FunctionTransformer(dequant, requant, check_inverse=True, validate=True),
        scaletf,
        FunctionTransformer(
            scipy.special.logit,
            scipy.special.expit,
            check_inverse=True,
            validate=True,
        ),
        StandardScaler(),
    ]
    return tfseq
