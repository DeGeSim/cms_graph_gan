from typing import List

import numpy as np
import torch
from torch_geometric.data import Data

from fgsim.utils.balanced_cluster import constrained_kmeans


def cluster_graph(
    graph: Data, branches: List[int], which_moon: np.ndarray
) -> List[torch.Tensor]:
    pointcloud = graph.x
    moons = [pointcloud[which_moon == 0], pointcloud[which_moon == 1]]

    brachings_lists = [
        [None for _ in range(len(moons))] for _ in range(len(branches) - 1)
    ]  # level # moon
    for imoon, points in enumerate(moons):
        for ilevel, cluster_size in enumerate(branches[::-1]):
            if len(points) == 1:
                continue
            assert len(points) % cluster_size == 0
            (Centroids, Members, f) = constrained_kmeans(
                points, [cluster_size for _ in range(len(points) // cluster_size)]
            )
            brachings_lists[ilevel][imoon] = Members
            points = Centroids
    brachings_lists_joined = []
    for ilevel in range(len(brachings_lists)):
        joined = np.hstack(
            (
                brachings_lists[ilevel][0],
                brachings_lists[ilevel][1] + max(brachings_lists[ilevel][0]) + 1,
            )
        )
        brachings_lists_joined.append(joined)
    brachings_lists_joined.append(np.array([0, 0]))
    return [torch.tensor(e).long() for e in brachings_lists_joined][::-1]
