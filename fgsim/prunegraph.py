# %%
import numpy as np
import networkx as nx

# %%
arr = np.array([[[1, 0], [1, 1]], [[1, 0], [0, 1]], [[0, 0], [1, 1]]])

graph = nx.grid_graph(dim=arr.shape)
# %%
