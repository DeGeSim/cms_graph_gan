from fgsim.io.sel_loader import Batch
from fgsim.ml.holder import Holder
from fgsim.models.metrics.dcd import cd


class LossGen:
    def __init__(self, lpnorm=2.0) -> None:
        self.lpnorm = lpnorm

    def __call__(
        self,
        holder: Holder,
        batch: Batch,
    ):
        # Loss of the generated samples

        n_features = batch.x.shape[1]
        batch_size = int(batch.batch[-1] + 1)
        loss = cd(
            holder.gen_points_w_grad.x.reshape(batch_size, -1, n_features),
            batch.x.reshape(batch_size, -1, n_features),
            lpnorm=self.lpnorm,
        ).mean()

        return loss
