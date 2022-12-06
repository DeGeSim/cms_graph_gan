import torch
from torch_geometric.data import Data


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.par = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, gen_batch: Data, *args, **kwargs):
        return self.par * torch.ones(
            (gen_batch.batch[-1] + 1, 1), device=gen_batch.x.device
        )
