import torch

from fgsim.config import device


def cuda_clear():
    if device.type == "cuda":
        torch.cuda.empty_cache()
