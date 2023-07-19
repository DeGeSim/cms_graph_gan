# %%
import numpy as np
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
from fgsim.loaders.calochallange.voxelize import dims, voxelize  # noqa: E402

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

if not torch.allclose(batch.x, batch_untf.x, rtol=1e-04):
    sel_tf = scaler.transfs_x[idx[1]]
    print(batch.x[idx], batch_untf.x[idx], idx[1])
    print(
        batch_scaled.x[idx], sel_tf.transform(batch_untf.x[:, idx[1]])[idx], idx[1]
    )

    tfsteps = [e[1] for e in sel_tf.steps]
    forward_res_step = [batch_untf.x[idx].numpy().reshape(-1, 1)]
    for _tf in tfsteps:
        forward_res_step.append(_tf.transform(forward_res_step[-1].copy()))
    backward_res_step = [forward_res_step[-1].copy()]
    for itf, _tf in enumerate(tfsteps[::-1]):
        a = backward_res_step[-1]
        b = forward_res_step[-1 - itf]
        print(a, b)
        if not np.allclose(a, b):
            print("the problem is ", tfsteps[-itf])
        backward_res_step.append(
            _tf.inverse_transform(backward_res_step[-1].copy())
        )


assert torch.allclose(batch.x, batch_untf.x, rtol=1e-04)
batch = postprocess(batch)
assert torch.allclose(batch.x, batch_untf.x)
pcs_out = voxelize(batch)
input_images = torch.stack([e[1].reshape(*dims) for e in pcs], 0)
assert torch.allclose(pcs_out, input_images)
# %%
