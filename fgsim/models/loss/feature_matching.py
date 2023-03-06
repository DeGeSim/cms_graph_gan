import torch
from torch.nn import MSELoss


class LossGen:
    def __init__(self) -> None:
        self.lossf = MSELoss()

    def __call__(
        self,
        d_gen_latftx: torch.Tensor,
        d_sim_latftx: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert d_gen_latftx.requires_grad
        assert not d_sim_latftx.requires_grad
        return self.lossf(d_gen_latftx, d_sim_latftx.detach())
