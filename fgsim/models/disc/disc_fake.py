import torch
from torch_geometric.data import Data


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.par = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, batch: Data):
        return self.par * torch.ones(batch.batch[-1] + 1, device=batch.x.device)
