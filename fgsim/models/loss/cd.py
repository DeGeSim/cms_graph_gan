from fgsim.io.sel_loader import Batch
from fgsim.ml.holder import Holder
from fgsim.models.metrics.dcd import cd


class LossGen:
    def __init__(
        self, lpnorm: float = 2.0, pow: float = 1.0, batch_wise: bool = False
    ) -> None:
        self.lpnorm = lpnorm
        self.batch_wise = batch_wise
        self.pow = pow

    def __call__(
        self,
        holder: Holder,
        batch: Batch,
    ):
        n_features = batch.x.shape[1]
        batch_size = int(batch.batch[-1] + 1)
        shape = (
            (1, -1, n_features) if self.batch_wise else (batch_size, -1, n_features)
        )
        loss = cd(
            holder.gen_points_w_grad.x.reshape(*shape),
            batch.x.reshape(*shape),
            lpnorm=self.lpnorm,
            pow=self.pow,
        ).mean()
        if holder.state.epoch >= 150:
            loss *= 0
        return loss
