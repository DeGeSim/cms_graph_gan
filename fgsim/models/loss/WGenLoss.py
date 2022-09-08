import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        d_gen: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert d_gen.requires_grad
        loss = d_gen.mean() * -1
        return loss
