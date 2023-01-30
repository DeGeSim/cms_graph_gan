import torch.nn as nn
import torch_geometric.nn as GravNetConv

# from torch_geometric.utils import to_dense_batch

# from fgsim.models.common.ffn import FFN

# https://proceedings.mlr.press/v80/achlioptas18a.html
# https://github.com/optas/latent_3d_points


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GravNetConv(
            in_channels=3,
            out_channels=1,
            space_dimensions=2,
            propagate_dimensions=3,
            k=10,
        )

    def forward(self, batch, cond):
        x = self.conv(batch.x, batch.batch)
        return x
