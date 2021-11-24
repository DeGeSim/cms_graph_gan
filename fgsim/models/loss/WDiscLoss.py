import torch


def loss(y: torch.Tensor, ypred: torch.Tensor):
    return -y.mean() + ypred.mean()
