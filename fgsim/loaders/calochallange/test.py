# %%
import numpy as np
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from fgsim.config import compute_conf, conf

compute_conf(conf, {"dataset_name": "calochallange", "command": "test"})


from fgsim.loaders.calochallange.objcol import (  # noqa: E402
    contruct_graph_from_row,
    file_manager,
    read_chunks,
    scaler,
)
from fgsim.loaders.calochallange.voxelize import (  # noqa: E402
    dims,
    sum_dublicate_hits,
    voxelize,
)

# %%
step = 1000
fpos = np.arange(0, 100_000, step)
for i in tqdm(fpos):
    pcs = read_chunks([(file_manager.files[0], i, i + step)])
    gl = []
    for pc in pcs:
        graph = contruct_graph_from_row(pc)
        graph.n_pointsv = (
            graph.y[..., conf.loader.y_features.index("num_particles")]
            .int()
            .reshape(-1)
        )
        graph.x = graph.x.double()
        graph.y = graph.y.double()

        gl.append(graph)

    batch = Batch.from_data_list(gl)
    batch_untf = batch.clone()
    batch.x = scaler.transform(batch.x, "x").double()
    batch.y = scaler.transform(batch.y, "y").double()

    batch_scaled = batch.clone()
    # alphapos = conf.loader.x_features.index("alpha")
    # batch.x[..., alphapos] = rotate_alpha(
    #     batch.x[..., alphapos].clone(), batch.batch, True
    # )
    assert (batch.x == batch_scaled.x).all()
    batch.x = scaler.inverse_transform(batch.x, "x").double()
    batch.y = scaler.inverse_transform(batch.y, "y").double()

    delta = (batch.x - batch_untf.x).abs()
    idx = torch.where(delta == delta.max())

    if not torch.allclose(batch.x, batch_untf.x, rtol=1e-04):
        sel_tf = scaler.transfs_x[idx[1]]
        print(batch.x[idx], batch_untf.x[idx], idx[1])
        print(
            batch_scaled.x[idx],
            sel_tf.transform(batch_untf.x[:, idx[1]]).squeeze()[idx[0]],
            idx[1],
        )

        tfsteps = [e[1] for e in sel_tf.steps]
        forward_res_step = [batch_untf.x[idx].numpy().reshape(-1, 1)]
        for _tf in tfsteps:
            forward_res_step.append(_tf.transform(forward_res_step[-1].copy()))

        backward_res_step = [batch_scaled.x[idx].numpy().reshape(-1, 1)]
        print(f"tranformed {backward_res_step[-1]}, scaled {forward_res_step[-1]}")
        for itf, _tf in enumerate(tfsteps[::-1]):
            backward_res_step.append(
                _tf.inverse_transform(backward_res_step[-1].copy())
            )
            a = backward_res_step[-1]
            b = forward_res_step[-2 - itf]
            print(a, b, "out of", _tf)

        raise Exception()

    assert torch.allclose(batch.x, batch_untf.x, rtol=1e-04)

    ## cc postprocess step by step

    # alphashift
    # alphas = batch.x[..., alphapos].clone()
    # shift = torch.randint(0, num_alpha, (batch.batch[-1] + 1,)).to(alphas.device)[
    #     batch.batch
    # ]
    # alphas = alphas.clone() + shift.float()
    # alphas[alphas > num_alpha - 1] -= num_alpha
    # batch.x[..., alphapos] = alphas

    batch = sum_dublicate_hits(batch)
    assert torch.allclose(batch.x, batch_untf.x)

    # batch = batch_to_Exyz(batch)
    # metrics: list[str] = conf.training.val.metrics
    # if "sphereratio" in metrics:
    #     batch["sphereratio"] = sphereratio(batch)
    # if "fpc" in metrics:
    #     batch["fpc"] = fpc_from_batch(batch)
    # if "showershape" in metrics:
    #     batch["showershape"] = analyze_layers(batch)
    # if "response" in metrics:
    #     batch["response"] = response(batch)

    assert torch.allclose(batch.x, batch_untf.x)
    pcs_out = voxelize(batch)
    input_images = torch.stack([e[1].reshape(*dims) for e in pcs], 0)
    assert torch.allclose(pcs_out.float(), input_images)
