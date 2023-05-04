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
        loss_list = []
        for ifeature, (g, s) in enumerate(zip(gen_latftx, sim_latftx)):
            assert g.requires_grad
            assert not s.requires_grad
            y = s.detach()
            mean, std = y.mean(0), y.std(0)
            std[std == 0] = 1
            yp = (y - mean) / (std + 1e-4)

            yhat = g
            yphat = (yhat - mean) / (std + 1e-4)
            loss = (yp - yphat).abs().mean()
            loss_list.append(loss.clone())
        return sum(loss_list)
