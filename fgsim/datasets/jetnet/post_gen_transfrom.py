import torch
from torch_scatter import scatter_add

from fgsim.config import conf
from fgsim.utils import check_tensor

from .scaler import scaler


def norm_pt_sum(pts, batchidx):
    pt_scaler = scaler.transfs_x[2]

    assert pt_scaler.method == "box-cox"
    assert pt_scaler.standardize
    # get parameters for the backward tranformation
    lmbd = pt_scaler.lambdas_[0]
    mean = pt_scaler._scaler.mean_[0]
    scale = pt_scaler._scaler.scale_[0]

    # Backwards transform
    #  pts_old = pts.clone()
    pts = pts.clone().double() * scale + mean
    #  pts_tf = pts.clone()

    idx_to_large = pts > 30

    if conf.command == "train":
        if idx_to_large.float().mean() > 0.1:
            raise Exception("To many points cant be scaled.")
    else:
        if idx_to_large.float().mean() > 0.01:
            raise Exception("To many points cant be scaled.")
    offset = pts[idx_to_large].detach() - 30
    pts[idx_to_large] = pts[idx_to_large] - offset

    check_tensor(pts)
    if lmbd == 0:
        pts = torch.exp(pts.clone())
    else:
        pts = torch.pow(pts.clone() * lmbd + 1, 1 / lmbd)
    check_tensor(pts)

    # Norm
    ptsum_per_batch = scatter_add(pts, batchidx, dim=-1)
    pts = pts / ptsum_per_batch[batchidx]
    check_tensor(pts)

    # Forward transform
    if lmbd == 0:
        pts = torch.log(pts.clone())
    else:
        pts = (torch.pow(pts.clone(), lmbd) - 1) / lmbd

    pts[idx_to_large] = pts[idx_to_large] + offset
    pts = (pts.clone() - mean) / scale
    check_tensor(pts)
    return pts.float()
