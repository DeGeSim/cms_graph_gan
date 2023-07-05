import torch
from scipy.stats import wasserstein_distance


def scaled_w1(s: torch.Tensor, g: torch.Tensor) -> float:
    assert s.dim() == g.dim() == 1
    s = s.detach().cpu()
    g = g.detach().cpu()
    mean, std = s.mean(), s.std()
    std[std == 0] = 1
    sp = (s - mean) / (std + 1e-4)
    gp = (g - mean) / (std + 1e-4)
    return float(wasserstein_distance(sp.numpy(), gp.numpy()))


def calc_cdf_dist(r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    dists = []
    assert r.shape == f.shape
    for iftx in range(r.shape[-1]):
        cdf_fake = f[..., iftx].sort()[0].cumsum(-1)
        cdf_real = r[..., iftx].sort()[0].cumsum(-1)

        cdf_fake /= cdf_fake[-1].clone()
        cdf_real /= cdf_real[-1].clone()

        cdfdist = (cdf_fake - cdf_real).abs().mean(0)
        dists.append(cdfdist)
    return torch.stack(dists, dim=-1)


def calc_cdf_and_weight(arr, weigths):
    val, sortidx = arr.sort()
    cdf = val.cumsum(-1)
    cdf /= cdf[-1].clone()
    w = weigths[sortidx]
    return cdf, w


def calc_wcdf_dist(
    r: torch.Tensor, f: torch.Tensor, rw: torch.Tensor, fw: torch.Tensor
) -> torch.Tensor:
    dists = []
    assert r.shape == f.shape
    for iftx in range(r.shape[-1]):
        cdf_fake, w_fake = calc_cdf_and_weight(f[..., iftx], fw)
        cdf_real, w_real = calc_cdf_and_weight(f[..., iftx], fw)

        ww = w_fake * w_real
        ww /= ww.sum()

        cdfdist = ((cdf_fake - cdf_real) * ww).abs().mean(0)
        dists.append(cdfdist)
    return torch.stack(dists, dim=-1)
