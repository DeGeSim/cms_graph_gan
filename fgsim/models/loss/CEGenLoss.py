import torch

from fgsim.config import conf, device


class LossGen:
    def __init__(
        self,
    ) -> None:
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )

    def __call__(
        self,
        d_gen: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert kwargs["gen_batch"].x.requires_grad
        assert d_gen.requires_grad
        errG = self.criterion(d_gen, torch.ones_like(d_gen))
        return errG
