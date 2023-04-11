import torch
from torch.nn import InstanceNorm1d, MSELoss


class LossGen:
    def __init__(self) -> None:
        self.lossf = MSELoss()
        #

    def __call__(
        self,
        gen_condreg: torch.Tensor,
        sim_batch: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_condreg.requires_grad
        if not hasattr(self, "norm"):
            self.norm = InstanceNorm1d(gen_condreg.shape[-1])
        l_gen_normed, l_sim_normed = (
            self.norm(e.mean(0).unsqueeze(0))
            for e in (gen_condreg, sim_batch.y.detach())
        )
        loss = self.lossf(l_gen_normed, l_sim_normed)

        return loss
