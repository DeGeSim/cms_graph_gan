# %%
import torch
from torch_geometric.data import Batch

from fgsim.config import compute_conf, conf

compute_conf(conf, {"dataset_name": "calochallange", "command": "test"})


from fgsim.loaders.calochallange.alpharot import rotate_alpha  # noqa: E402
from fgsim.loaders.calochallange.objcol import (  # noqa: E402
    contruct_graph_from_row,
    file_manager,
    read_chunks,
    scaler,
)
from fgsim.loaders.calochallange.postprocess import postprocess  # noqa: E402
from fgsim.loaders.calochallange.voxelize import voxelize  # noqa: E402

# %%
pcs = read_chunks([(file_manager.files[0], 0, 3)])
gl = []
for pc in pcs:
    graph = contruct_graph_from_row(pc)
    graph.n_pointsv = (
        graph.y[..., conf.loader.y_features.index("num_particles")]
        .int()
        .reshape(-1)
    )

    gl.append(graph)


batch = Batch.from_data_list(gl)
batch_untf = batch.clone()
batch.x = scaler.transform(batch.x, "x")
batch.y = scaler.transform(batch.y, "y")

batch_scaled = batch.clone()
alphapos = conf.loader.x_features.index("alpha")
batch.x[..., alphapos] = rotate_alpha(
    batch.x[..., alphapos].clone(), batch.batch, True
)
assert (batch.x == batch_scaled.x).all()
batch.x = scaler.inverse_transform(batch.x, "x")
batch.y = scaler.inverse_transform(batch.y, "y")

delta = (batch.x - batch_untf.x).abs()
idx = torch.where(delta == delta.max())
print(batch.x[idx[0], :], batch_untf.x[idx[0], :], idx[1])


assert torch.allclose(batch.x, batch_untf.x, rtol=1e-04)
batch = postprocess(batch)
assert (batch.x == batch_untf.x).all()
pcs_out = voxelize(batch)
assert pcs_out == torch.stack([torch.tensor(e[1]) for e in pcs], 0)
# %%
