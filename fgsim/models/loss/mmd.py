from typing import List

import torch

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float, kernel, bandwidth: List[float]) -> None:
        self.factor = factor
        self.kernel = kernel
        self.bandwidth = bandwidth

    def __call__(self, holder: Holder, batch: Batch, *args, **kwargs):
        shape = (
            conf.loader.batch_size,
            conf.loader.max_points * conf.loader.n_features,
        )
        sim_sample = batch.x.reshape(*shape)
        gen_sample = holder.gen_points_w_grad.x.reshape(*shape)
        assert sim_sample.shape == gen_sample.shape

        loss = MMD(
            sim_sample,
            gen_sample,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        )
        loss.backward(retain_graph=True)
        return float(loss)


def MMD(x, y, bandwidth, kernel):
    # https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
    """Emprical maximum mean discrepancy. The lower the result
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    for a in bandwidth:
        if kernel == "multiscale":
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1
        elif kernel == "rbf":
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
        else:
            raise Exception

    return torch.mean(XX + YY - 2.0 * XY)
