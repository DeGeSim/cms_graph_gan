import torch


def loss(ytrue: torch.Tensor, ypred: torch.Tensor):
    return -ytrue.mean() + ypred.mean()
