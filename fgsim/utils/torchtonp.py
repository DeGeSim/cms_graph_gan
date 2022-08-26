from typing import Callable

import numpy as np
import torch


def convert_to_torch(e):
    if isinstance(e, np.ndarray):
        return torch.from_numpy(e)
    else:
        return e


def wrap_torch_to_np(fct: Callable):
    def inner(*args, **kwargs):
        convert = False
        args = list(args)
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor):
                args[i] = args[i].cpu().numpy()
                convert = True
        for k in kwargs:
            if isinstance(kwargs[k], torch.Tensor):
                kwargs[k] = kwargs[k].cpu().numpy()
                convert = True
        res = fct(*args, **kwargs)
        if convert:
            if isinstance(res, tuple):
                return tuple(convert_to_torch(e) for e in res)
            else:
                return convert_to_torch(res)
        else:
            return res

    return inner
