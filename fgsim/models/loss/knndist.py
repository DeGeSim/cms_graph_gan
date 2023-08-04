import torch
from caloutils.distances import scale_b_to_a, wmse
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, knn_graph

from fgsim.config import conf
from fgsim.io.sel_loader import scaler

eidx = conf.loader.x_ftx_energy_pos


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_batch: Batch,
        sim_batch: Batch,
        **kwargs,
    ):
        assert gen_batch.x.requires_grad

        posidx = [i for i in range(gen_batch.x.shape[-1]) if i != eidx]

        loss = torch.tensor(0.0).to(gen_batch.x.device)
        for fridx in [[eidx], posidx]:
            nndelta_sim = self.nndist(sim_batch, fridx)
            nndelta_gen = self.nndist(gen_batch, fridx)
            nnd_sim_scaled, nnd_gen_scaled = scale_b_to_a(nndelta_sim, nndelta_gen)
            if fridx == [eidx]:
                loss += wmse(nnd_sim_scaled, nnd_gen_scaled)
            else:
                #  sw = self.inv_scale_hitE(sim_batch).to(dev)
                #  gw = self.inv_scale_hitE(gen_batch).to(dev)

                #  assert (sw > 0).all() and (gw > 0).all()
                #  loss += wmse(nnd_sim_scaled, nnd_gen_scaled, sw, gw)
                loss += wmse(nnd_sim_scaled, nnd_gen_scaled)

        return loss

    def inv_scale_hitE(self, batch):
        return torch.tensor(
            scaler.transfs_x[eidx].inverse_transform(
                batch.x[:, [eidx]].detach().cpu().numpy()
            )
        ).squeeze()

    def nndist(self, batch, slice):
        x = batch.x[:, slice]
        batchidx = batch.batch
        ei = knn_graph(x.clone(), k=3, batch=batchidx, loop=False)
        delta = (x[ei[0]] - x[ei[1]]).abs().mean(1)
        delta_aggr = global_add_pool(delta, ei[1])
        return delta_aggr
