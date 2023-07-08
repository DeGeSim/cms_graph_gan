from typing import Optional

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def sw1(s: torch.Tensor, g: torch.Tensor, **kwargs) -> np.ndarray:
    assert s.dim() == g.dim() == 1
    s = s.detach().cpu()
    g = g.detach().cpu()
    mean, std = s.mean(), s.std()
    std[std == 0] = 1
    sp = (s - mean) / (std + 1e-4)
    gp = (g - mean) / (std + 1e-4)
    return wasserstein_distance(sp.numpy(), gp.numpy())


def calc_scaled1d_dist(r: torch.Tensor, f: torch.Tensor, **kwargs) -> np.ndarray:
    dists = []
    assert r.shape == f.shape
    for iftx in range(r.shape[-1]):
        cdfdist = sw1(r[..., iftx], f[..., iftx])
        dists.append(cdfdist)
    return np.stack(dists, axis=0)


def cdf(arr):
    val, _ = arr.sort()
    cdf = val.cumsum(-1)
    cdf /= cdf[-1].clone()
    return cdf


def calc_cdf_dist(r: torch.Tensor, f: torch.Tensor, **kwargs) -> np.ndarray:
    dists = []
    assert r.shape == f.shape
    for iftx in range(r.shape[-1]):
        cdfdist = (cdf(f[..., iftx]) - cdf(r[..., iftx])).abs().mean(0)
        dists.append(cdfdist)
    return torch.stack(dists, dim=0).cpu().numpy()


def wcdf(arr, weigths):
    val, sortidx = arr.sort()
    cdf = val.cumsum(-1)
    cdf /= cdf[-1].clone()
    w = weigths[sortidx]
    return cdf, w


def calc_wcdf_dist(
    r: torch.Tensor, f: torch.Tensor, rw: torch.Tensor, fw: torch.Tensor, **kwargs
) -> np.ndarray:
    dists = []
    assert r.shape == f.shape
    for iftx in range(r.shape[-1]):
        cdf_real, w_real = wcdf(r[..., iftx], rw)
        cdf_fake, w_fake = wcdf(f[..., iftx], fw)

        ww = w_fake * w_real
        ww /= ww.sum()

        cdfdist = ((cdf_fake - cdf_real) * ww).abs().mean(0)
        dists.append(cdfdist)
    return torch.stack(dists, dim=0).cpu().numpy()


def cdf_by_hist(
    arr: torch.Tensor, bins: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    val = torch.histogram(arr, bins=bins, weight=weight)
    cdf = val.hist.cumsum(-1)
    cdf /= cdf[-1].clone()
    return cdf


def calc_hist_dist(
    r: torch.Tensor,
    f: torch.Tensor,
    bins: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> np.ndarray:
    dists = []
    assert r.shape == f.shape
    if len(r.shape) == 1:
        r = r.unsqueeze(-1)
        f = f.unsqueeze(-1)
    for iftx in range(r.shape[-1]):
        cdf_r = cdf_by_hist(r[..., iftx], bins[iftx], rw)
        cdf_f = cdf_by_hist(f[..., iftx], bins[iftx], fw)

        dist = (cdf_f - cdf_r).abs().mean()

        dists.append(dist)
    return torch.stack(dists, dim=0).cpu().numpy()
