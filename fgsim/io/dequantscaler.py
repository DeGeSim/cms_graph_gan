from functools import partial

import numpy as np
import scipy
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .idxscale import IdxToScale


def dequant(x):
    noise = np.random.rand(*x.shape)
    return x.astype("float") + noise


def requant(x):
    x_copy = x.copy()
    edgecase = (x.astype("int") == x).squeeze()
    x[edgecase] -= 1
    x[~edgecase] = np.floor(x[~edgecase])
    assert np.all(x.astype("int") == x)
    delta = x_copy - x
    assert (delta >= 0).all() and (delta <= 1).all()
    return x


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
