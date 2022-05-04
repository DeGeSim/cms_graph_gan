#
# Author: Stanislaw Adaszewski, 2015
#
# Bradley PS, Bennett KP, Demiriz A (2000) Constrained K-Means Clustering.
# Microsoft Research. Available: http://research.microsoft.com/pubs/69796/tr-2000-65.pdf
import networkx as nx
import numpy as np


def constrained_kmeans(data, cluster_sizes, maxiter=None, fixedprec=1e9):
    data = np.array(data)

    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)

    C = min_ + np.random.random((len(cluster_sizes), data.shape[1])) * (max_ - min_)
    M = np.array([-1] * len(data), dtype=int)

    itercnt = 0
    while True:
        itercnt += 1

        # memberships
        g = nx.DiGraph()
        g.add_nodes_from(range(0, data.shape[0]), demand=-1)  # points
        for i in range(0, len(C)):
            g.add_node(len(data) + i, demand=cluster_sizes[i])

        # Calculating cost...
        cost = np.array(
            [
                np.linalg.norm(
                    np.tile(data.T, len(C)).T
                    - np.tile(C, len(data)).reshape(len(C) * len(data), C.shape[1]),
                    axis=1,
                )
            ]
        )
        # Preparing data_to_C_edges...
        data_to_C_edges = np.concatenate(
            (
                np.tile([range(0, data.shape[0])], len(C)).T,
                np.tile(
                    np.array([range(data.shape[0], data.shape[0] + C.shape[0])]).T,
                    len(data),
                ).reshape(len(C) * len(data), 1),
                cost.T * fixedprec,
            ),
            axis=1,
        ).astype(np.uint64)
        # Adding to graph
        g.add_weighted_edges_from(data_to_C_edges)

        a = len(data) + len(C)
        g.add_node(a, demand=len(data) - np.sum(cluster_sizes))
        C_to_a_edges = np.concatenate(
            (
                np.array([range(len(data), len(data) + len(C))]).T,
                np.tile([[a]], len(C)).T,
            ),
            axis=1,
        )
        g.add_edges_from(C_to_a_edges)

        # Calculating min cost flow...
        f = nx.min_cost_flow(g)

        # assign
        M_new = np.ones(len(data), dtype=int) * -1
        for i in range(len(data)):
            p = sorted(iter(f[i].items()), key=lambda x: x[1])[-1][0]
            M_new[i] = p - len(data)

        # stop condition
        if np.all(M_new == M):
            # Stop
            return (C, M, f)

        M = M_new

        # compute new centers
        for i in range(len(C)):
            C[i, :] = np.mean(data[M == i, :], axis=0)

        if maxiter is not None and itercnt >= maxiter:
            # Max iterations reached
            return (C, M, f)


# from sklearn import cluster

# res = cluster.DBSCAN(
#     eps=1e-10, min_samples=1, algorithm="ball_tree", metric="haversine"
# ).fit(np.radians(moon))
# res = cluster.Birch(threshold=1, n_clusters=4, branching_factor=2).fit(
#     np.radians(moon)
# )
# val, counts = np.unique(res, return_counts=True)
# if np.all(counts == counts[0]):
#     print(f" {counts} 👍🏽")
# else:
#     print(f" {counts} 😭")
