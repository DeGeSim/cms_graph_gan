import torch
from torch.nn import InstanceNorm1d, MSELoss


class LossGen:
    def __init__(self) -> None:
        self.lossf = MSELoss()
        #

    def __call__(
        self,
        d_gen_latftx: torch.Tensor,
        d_sim_latftx: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert d_gen_latftx.requires_grad
        assert not d_sim_latftx.requires_grad
        if not hasattr(self, "norm"):
            self.norm = InstanceNorm1d(d_gen_latftx.shape[-1])
        l_gen_normed, l_sim_normed = (
            self.norm(e.mean(0).unsqueeze(0))
            for e in (d_gen_latftx, d_sim_latftx.detach())
        )
        loss = self.lossf(l_gen_normed, l_sim_normed)

        return loss
