# https://github.com/rkansal47/MPGAN/blob/development/metrics/metrics.py
# based on https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py

import numpy as np
from numba import njit, prange, set_num_threads
from numpy.typing import ArrayLike
from scipy.stats import iqr


def normalise_features(X: ArrayLike, Y: ArrayLike = None):
    maxes = np.max(np.abs(X), axis=0)

    return (X / maxes, Y / maxes) if Y is not None else X / maxes


@njit
def _mmd_quadratic_unbiased(XX: ArrayLike, YY: ArrayLike, XY: ArrayLike):
    m, n = XX.shape[0], YY.shape[0]
    # subtract diagonal 1s
    return (
        (XX.sum() - np.trace(XX)) / (m * (m - 1))
        + (YY.sum() - np.trace(YY)) / (n * (n - 1))
        - 2 * np.mean(XY)
    )


@njit
def _poly_kernel_pairwise(X: ArrayLike, Y: ArrayLike, degree: int) -> np.ndarray:
    gamma = 1.0 / X.shape[-1]
    return (X @ Y.T * gamma + 1.0) ** degree


@njit
def mmd_poly_quadratic_unbiased(
    X: ArrayLike, Y: ArrayLike, degree: int = 4
) -> float:
    XX = _poly_kernel_pairwise(X, X, degree=degree)
    YY = _poly_kernel_pairwise(Y, Y, degree=degree)
    XY = _poly_kernel_pairwise(X, Y, degree=degree)
    return _mmd_quadratic_unbiased(XX, YY, XY)


@njit(parallel=True)
def _average_batches_mmd(X, Y, num_batches, batch_size, seed):
    # can't use list.append with numba prange
    # https://github.com/numba/numba/issues/4206#issuecomment-503947050
    vals_point = np.zeros(num_batches, dtype=np.float64)
    for i in prange(num_batches):
        np.random.seed(seed + i * 1000)  # in case of multi-threading
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = mmd_poly_quadratic_unbiased(rand_sample1, rand_sample2, degree=4)
        vals_point[i] = val

    return vals_point


def mmd(
    X: ArrayLike,
    Y: ArrayLike,
    num_batches: int = 10,
    batch_size: int = 5000,
    seed: int = 42,
    normalise: bool = True,
    num_threads: int = 6,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    set_num_threads(num_threads)
    vals_point = _average_batches_mmd(X, Y, num_batches, batch_size, seed)
    return [np.median(vals_point), iqr(vals_point, rng=(16.275, 83.725))]


class Metric:
    def __init__(
        self,
    ):
        pass

    def __call__(self, sim_efps, gen_efps, **kwargs) -> tuple[float, float]:
        if np.isnan(gen_efps).any():
            return (1e5, 1e5)
        score = mmd(sim_efps, gen_efps)
        return tuple(min(float(e), 1e5) for e in score)


kpd = Metric()
