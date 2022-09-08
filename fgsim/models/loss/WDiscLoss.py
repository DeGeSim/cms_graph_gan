import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        d_sim: torch.Tensor,
        d_gen: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert d_sim.requires_grad and d_sim.requires_grad
        return d_gen.mean() - d_sim.mean()
