# %%
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.pyplot import figure

# %%
with h5py.File("wd/forward/Ele_FixedAngle/EleEscan_1_1.h5") as f:
    excalimg = f["ECAL"][:]
# %%
componentsL = []
size_fractionL = []
energyL = []
energy_fractionL = []
largest_graph_energy_fraction = []
nnodes = []
# %%

figure(figsize=(5, 5), dpi=300)
a = np.swapaxes(excalimg[0], 0, 2)
plt.imshow(a[15] != 0)
plt.savefig("/home/mscham/occupation_pruning.pdf")
# %%
for i in range(100):
    a = np.swapaxes(excalimg[i], 0, 2)

    # https://stackoverflow.com/questions/62671695/convert-0-1-matrix-to-a-2d-grid-graph-in-python
    G = nx.grid_graph(a.shape[::-1])

    # remove those nodes where the corresponding value is != 0
    for val, node in zip(a.ravel(), sorted(G.nodes())):
        if val == 0.0:
            G.remove_node(node)
    nnodes.append(len(G.nodes))

    components = list(
        nx.connected_components(G)
    )  # list because it returns a generator
    complensL = [len(e) for e in components]
    componentsL = componentsL + complensL
    size_fractionL.append(np.max(complensL) / np.sum(complensL))

    eL = []
    for subgraph in components:
        s = 0
        for node in subgraph:
            s = s + a[node]
        eL.append(s)
    # eL = [sum([a[node] for node in subgraph]) for subgraph in components]
    energyL = energyL + eL
    energy_fractionL.append(np.max(eL) / np.sum(eL))

# %%

arr = np.array(G.nodes)
for idim in range(len(a.shape)):
    print(f"adim {a.shape[idim]} nodesmax {max(arr[:,idim])}")


# %%
aflat = a.flatten()
plt.hist(aflat[aflat < 2], bins=100)
plt.yscale("log")


# %%
plt.hist(componentsL)
plt.title("Histogram of the Size of the subgraphs")
plt.suptitle(
    "Average size fraction for the largest subgraph:"
    + f" {np.round(np.mean(size_fractionL)*100)}%\n"
)
plt.yscale("log")
# %%

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches((15, 5))

ax1.hist(energyL)
ax1.set_title("Histogram of the energy of the subgraphs")
ax1.set_yscale("log")
plt.text(
    1,
    1,
    "Average energy fraction\n for the "
    + f"largest subgraph {np.round(np.mean(energy_fractionL)*100)}%",
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax1.transAxes,
    size=13,
)
ax2.hist(componentsL)
ax2.set_title("Histogram of the Size of the subgraphs")

plt.text(
    1,
    1,
    "Average size fraction\n for the "
    + f"largest subgraph: {np.round(np.mean(size_fractionL)*100)}%",
    horizontalalignment="right",
    verticalalignment="top",
    transform=ax2.transAxes,
    size=13,
)
ax2.set_yscale("log")
plt.savefig("/home/mscham/res_graph_pruning.pdf")
# plt.tight_layout()

# %%


def get_layer_subgraph(G, layer):
    assert 0 <= layer < a.shape[0]
    subG = G.subgraph([node for node in G.nodes() if node[0] == layer])
    return subG


# %%

plt.figure(figsize=(9, 9))
# coordinate rotation
pos = {(x, y, z): (y, -x, z) for x, y, z in G.nodes()}
nx.draw(G, pos=pos, node_color="grey", width=4, node_size=4)
# %%
