import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def hinge_act(self, x: torch.Tensor):
        return (x < 0).float() * x

    def __call__(
        self,
        d_sim: torch.Tensor,
        d_gen: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert d_sim.requires_grad and d_sim.requires_grad
        loss = -self.hinge_act(d_sim - 1).mean()
        loss += -self.hinge_act(-d_gen - 1).mean()
        return loss
