import torch


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_latftx: torch.Tensor,
        sim_latftx: torch.Tensor,
        **kwargs,
    ):
        assert kwargs["gen_batch"].x.requires_grad
        assert gen_latftx.requires_grad
        assert not sim_latftx.requires_grad
        y = sim_latftx.detach()
        mean, std = y.mean(0), y.std(0)
        std[std == 0] = 1
        yp = (y - mean) / (std + 1e-3)

        yhat = gen_latftx
        yphat = (yhat - mean) / (std + 1e-3)
        loss = (yp - yphat).abs().mean()
        return loss
