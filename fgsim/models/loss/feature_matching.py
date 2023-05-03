import torch
from torch.nn import InstanceNorm1d, MSELoss


class LossGen:
    def __init__(self) -> None:
        self.lossf = MSELoss()
        self.norms = []

    def __call__(
        self,
        gen_latftx: list[torch.Tensor],
        sim_latftx: list[torch.Tensor],
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        loss_list = []
        for ifeature, (g, s) in enumerate(zip(gen_latftx, sim_latftx)):
            assert g.requires_grad
            assert not s.requires_grad
            if ifeature == len(self.norms):
                self.norms.append(InstanceNorm1d(g.shape[-1]))
            l_gen_normed, l_sim_normed = (
                self.norms[ifeature](e) for e in (g, s.detach())
            )
            loss_list.append(self.lossf(l_gen_normed, l_sim_normed))

        return sum(loss_list)
