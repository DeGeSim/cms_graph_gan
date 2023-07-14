import torch

from fgsim.config import conf
from fgsim.utils import check_tensor

from .objcol import scaler


def rotate_alpha(pts, batchidx):
    batch_size = int(batchidx[-1] + 1)
    alphapos = conf.loader.x_features.index("alpha")
    ascalers = scaler.transfs_x[alphapos].steps[::-1]

    assert ascalers[0][0] == "standardscaler"
    mean = ascalers[0][1].mean_[0]
    scale = ascalers[0][1].scale_[0]

    # Backwards transform #0 stdscalar
    pts = pts.clone().double() * scale + mean

    # Backwards transform #1 logit
    pts = torch.special.expit(pts)

    # Rotation
    # smin, smax = ascalers[2][1].feature_range
    shift = torch.rand(batch_size).to(pts.device)[batchidx]
    pts = pts.clone() + shift
    pts[pts > 1] -= 1

    # Forward transform #1 logit
    pts = torch.special.logit(pts.clone())

    # Forward transform #0 stdscalar
    pts = (pts.clone() - mean) / scale
    check_tensor(pts)
    return pts.float()
