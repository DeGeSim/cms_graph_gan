import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def hinge_act(self, x: torch.Tensor):
        return (x < 0).float() * x

    def __call__(
        self,
        sim_crit: torch.Tensor,
        gen_crit: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert sim_crit.requires_grad and sim_crit.requires_grad
        loss = -self.hinge_act(sim_crit - 1).mean()
        loss += -self.hinge_act(-gen_crit - 1).mean()
        return loss
