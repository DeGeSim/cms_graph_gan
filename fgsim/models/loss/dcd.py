from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder
from fgsim.models.metrics.dcd import dcd


class LossGen:
    def __init__(self, alpha, lpnorm=2.0) -> None:
        self.alpha = alpha
        self.lpnorm = lpnorm

    def __call__(self, holder: Holder, batch: Batch):
        # Loss of the generated samples

        n_features = batch.x.shape[1]
        batch_size = int(batch.batch[-1] + 1)
        loss = dcd(
            holder.gen_points_w_grad.x.reshape(batch_size, -1, n_features),
            batch.x.reshape(batch_size, -1, n_features),
            alpha=self.alpha,
            lpnorm=self.lpnorm,
        ).mean()

        return loss
