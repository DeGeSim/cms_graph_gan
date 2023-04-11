import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_crit: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_crit.requires_grad
        loss = -1 * gen_crit.mean()
        return loss
