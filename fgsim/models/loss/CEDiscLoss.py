import torch


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self) -> None:
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(
        self,
        d_sim: torch.Tensor,
        d_gen: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert not kwargs["gen_batch"].x.requires_grad
        assert d_sim.requires_grad and d_sim.requires_grad
        sample_disc_loss = self.criterion(d_sim, torch.ones_like(d_sim))
        gen_disc_loss = self.criterion(d_gen, torch.zeros_like(d_gen))

        return gen_disc_loss + sample_disc_loss
