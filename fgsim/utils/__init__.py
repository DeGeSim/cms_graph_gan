import torch

from .model_summary import log_model


def check_tensor(*arrs: torch.Tensor):
    for iarr, x in enumerate(arrs):
        if not x.isfinite().all():
            raise RuntimeError(f"Tensor {iarr} NaN or infinite")
