import torch


class LossGen:
    def __init__(self) -> None:
        self.lossf = torch.nn.MSELoss()

    def __call__(
        self,
        d_gen: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert kwargs["gen_batch"].x.requires_grad
        assert d_gen.requires_grad
        return self.lossf(d_gen, torch.ones_like(d_gen))
