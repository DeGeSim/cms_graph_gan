from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder
from fgsim.models.metrics.dcd import dcd


class LossGen:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, holder: Holder, batch: Batch):
        # Loss of the generated samples

        n_features = batch.x.shape[1]
        batch_size = int(batch.batch[-1] + 1)
        loss = self.factor * dcd(
            holder.gen_points_w_grad.x.reshape(batch_size, -1, n_features),
            batch.x.reshape(batch_size, -1, n_features),
        )
        loss.backward(retain_graph=True)

        return float(loss)
