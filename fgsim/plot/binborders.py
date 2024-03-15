import numpy as np


def binborders_wo_outliers(points: np.ndarray, bins=50, cut_qt=0.05) -> np.ndarray:
    assert len(points.shape) == 1
    if len(points) > 10_000:
        points = np.random.choice(points, 10_000)
    points = points[~np.isnan(points)]

    if len(np.unique(points[:500])) < 50:
        uniques = np.unique(points)
        if len(uniques) < 50:
            delta = (uniques[1] - uniques[0]) / 2
            return np.concatenate([uniques[0:1] - delta, uniques + delta])

    return np.linspace(
        -upper_limit(-points, cut_higher=1 - cut_qt, cut_lower=1 - cut_qt * 4),
        upper_limit(points, cut_higher=1 - cut_qt, cut_lower=1 - cut_qt * 4),
        num=bins,
        endpoint=True,
    )


def upper_limit(points: np.ndarray, cut_higher=0.95, cut_lower=0.8) -> float:
    assert 1 >= cut_higher > cut_lower > 0
    higher, lower = np.quantile(points, [cut_higher, cut_lower])
    delta = higher - lower

    lexpol = lower + delta * (cut_higher / cut_lower) * 2
    return min(lexpol, np.max(points))


def bincenters(bins: np.ndarray) -> np.ndarray:
    return (bins[1:] + bins[:-1]) / 2


def bounds_wo_outliers(points: np.ndarray) -> tuple:
    median = np.median(points, axis=0)

    # med_abs_lfluk = np.sqrt(np.mean((points[points < median] - median) ** 2))
    # med_abs_ufluk = np.sqrt(np.mean((points[points > median] - median) ** 2))
    # upper = median + max(med_abs_ufluk,med_abs_ufluk)
    # lower = median - max(med_abs_ufluk,med_abs_ufluk)
    outlier_scale = (
        max(
            np.abs(np.quantile(points, 0.99) - median),
            np.abs(np.quantile(points, 0.01) - median),
        )
        * 1.1
    )
    upper = median + outlier_scale
    lower = median - outlier_scale
    # print(lower,np.min(points), upper,np.max(points))
    upper = np.min([upper, np.max(points)])
    lower = np.max([lower, np.min(points)])
    return lower, upper


def chip_to_binborders(arr, binborders):
    eps = (binborders[-1] - binborders[0]) / 1e6
    return np.clip(arr, binborders[0] + eps, binborders[-1] - eps)
