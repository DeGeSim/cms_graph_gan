import torch


class LossGen:
    def __init__(self) -> None:
        self.lossf = torch.nn.MSELoss()

    def __call__(
        self,
        d_sim: torch.Tensor,
        d_gen: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert d_sim.requires_grad and d_sim.requires_grad
        return self.lossf(torch.ones_like(d_sim), d_sim) + self.lossf(
            torch.zeros_like(d_gen), d_gen
        )
