# https://github.com/rkansal47/MPGAN/blob/development/metrics/metrics.py
# based on https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit

from fgsim.monitoring import logger


def linear(x, intercept, slope):
    return intercept + slope * x


def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            f"fid calculation produces singular product; adding {eps} to diagonal"
            " of cov estimates"
        )
        logger.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        # raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def frechet_gaussian_distance(X, Y, normalise: bool = True) -> float:
    if normalise:
        X, Y = normalise_features(X, Y)

    mu1 = np.mean(X, axis=0)
    sigma1 = np.cov(X, rowvar=False)
    mu2 = np.mean(Y, axis=0)
    sigma2 = np.cov(Y, rowvar=False)

    return _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def normalise_features(X, Y=None):
    maxes = np.max(np.abs(X), axis=0)

    return (X / maxes, Y / maxes) if Y is not None else X / maxes


def fpd_infinity(
    X,
    Y,
    min_samples: int = 5_000,
    max_samples: int = 50_000,
    num_batches: int = 10,
    num_points: int = 200,
    seed: int = 42,
    normalise: bool = True,
    n_jobs=None,
):
    if normalise:
        X, Y = normalise_features(X, Y)

    # Choose the number of images to evaluate FID_N at regular intervals over N
    batches = (
        1 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)
    ).astype("int32")
    # batches = np.linspace(min_samples, max_samples, num_points).astype("int32")

    np.random.seed(seed)

    vals = []

    for i, batch_size in enumerate(batches):
        vals_point = []
        for _ in range(num_batches):
            rand1 = np.random.choice(len(X), size=batch_size)
            rand2 = np.random.choice(len(Y), size=batch_size)

            rand_sample1 = X[rand1]
            rand_sample2 = Y[rand2]

            val = frechet_gaussian_distance(
                rand_sample1, rand_sample2, normalise=False
            )
            vals_point.append(val)

        vals.append(np.mean(vals_point))

    vals = np.array(vals)

    params, covs = curve_fit(
        linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf])
    )

    return (params[0], np.sqrt(np.diag(covs)[0]))


class Metric:
    def __init__(
        self,
    ):
        pass

    def __call__(self, sim_efps, gen_efps, **kwargs) -> tuple[float, float]:
        if np.isnan(gen_efps).any():
            return (1e5, 1e5)
        score = fpd_infinity(sim_efps, gen_efps)
        return tuple(min(float(e), 1e5) for e in score)


fgd = Metric()
