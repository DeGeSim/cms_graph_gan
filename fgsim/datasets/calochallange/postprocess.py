import torch
from caloutils import calorimeter
from caloutils.processing import shift_sum_multi_hits
from torch_geometric.data import Batch

from fgsim.config import conf

from .convcoord import batch_to_Exyz
from .pca import fpc_from_batch
from .shower import analyze_layers, cyratio, response, sphereratio


def postprocess(batch: Batch, sim_or_gen: str) -> Batch:
    alphapos = conf.loader.x_features.index("alpha")
    num_alpha = calorimeter.num_alpha

    batch = shift_sum_multi_hits(batch, forbid_dublicates=sim_or_gen == "sim")
    if sim_or_gen == "gen":
        alphas = batch.x[..., alphapos].clone()

        shift = torch.randint(0, num_alpha, (batch.batch[-1] + 1,)).to(
            alphas.device
        )[batch.batch]
        alphas = alphas.clone() + shift.float()
        alphas[alphas > num_alpha - 1] -= num_alpha

        batch.x[..., alphapos] = alphas

    batch = batch_to_Exyz(batch)
    metrics: list[str] = conf.training.val.metrics
    if "sphereratio" in metrics:
        batch["sphereratio"] = sphereratio(batch)
    if "cyratio" in metrics:
        batch["cyratio"] = cyratio(batch)
    if "fpc" in metrics:
        batch["fpc"] = fpc_from_batch(batch)
    if "showershape" in metrics:
        batch["showershape"] = analyze_layers(batch)
    if "response" in metrics:
        batch["response"] = response(batch)

    return batch
