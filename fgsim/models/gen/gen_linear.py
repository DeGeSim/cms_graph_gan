import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

# simple linear model from
# https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook


class ModelClass(nn.Module):
    def __init__(self, random_size, n_points, n_features, batch_size):
        super().__init__()
        self.n_points = n_points
        self.batch_size = batch_size
        self.n_features = n_features
        self.z_shape = batch_size, 1, random_size
        self.fc1 = nn.Linear(random_size, 1000)
        self.fc2 = nn.Linear(1000, 600)
        self.fc3 = nn.Linear(600, 1000)
        self.fc4 = nn.Linear(1000, n_points * n_features)

    def forward(self, samples):
        x = torch.sigmoid(self.fc1(samples))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        x = x.reshape(
            self.batch_size,
            self.n_points,
            self.n_features,
        )
        return Batch.from_data_list([Data(x=e) for e in x])
