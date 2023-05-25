import numpy as np
import scipy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .idxscale import IdxToScale


def dequant(x):
    noise = np.random.rand(*x.shape)
    return x + noise


def requant(x):
    return np.floor(x)


def dequant_stdscale():
    return make_pipeline(
        FunctionTransformer(dequant, requant, check_inverse=True),
        IdxToScale((0, 1)),
        FunctionTransformer(
            scipy.special.logit, scipy.special.expit, check_inverse=True
        ),
        StandardScaler(),
    )
