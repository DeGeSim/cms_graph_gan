from typing import Optional

import torch


def loss(ytrue: Optional[torch.Tensor], ypred: torch.Tensor):
    return -ypred.mean()


def LossGen():
    return loss
