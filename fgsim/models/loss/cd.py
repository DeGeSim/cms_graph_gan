from torch_geometric.data import Batch

from fgsim.models.metrics.dcd import cd


class LossGen:
    def __init__(
        self, lpnorm: float = 2.0, pow: float = 1.0, batch_wise: bool = False
    ) -> None:
        self.lpnorm: float = lpnorm
        self.batch_wise: bool = batch_wise
        self.pow: float = pow

    def __call__(
        self,
        sim_batch: Batch,
        gen_batch: Batch,
        **kwargs,
    ):
        assert gen_batch.x.requires_grad
        n_features = sim_batch.x.shape[1]
        batch_size = int(sim_batch.batch[-1] + 1)
        shape = (
            (1, -1, n_features) if self.batch_wise else (batch_size, -1, n_features)
        )
        loss = cd(
            gen_batch.x.reshape(*shape),
            sim_batch.x.reshape(*shape),
            lpnorm=self.lpnorm,
            pow=self.pow,
        ).mean()
        # if holder.state['epoch'] >= 150:
        #     loss *= 0
        if loss < 0:
            raise Exception
        return loss
