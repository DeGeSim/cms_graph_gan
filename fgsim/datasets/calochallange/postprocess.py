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

    batch = shift_sum_multi_hits(batch, forbid_dublicates=False)
    if sim_or_gen == "gen":
        alphas = batch.x[..., alphapos].clone()

        shift = torch.randint(0, num_alpha, (batch.batch[-1] + 1,)).to(
            alphas.device
        )[batch.batch]
        alphas = alphas.clone() + shift.float()
        alphas[alphas > num_alpha - 1] -= num_alpha

        batch.x[..., alphapos] = alphas

    batch = batch_to_Exyz(batch)
    metrics: list[str]

    match conf.command:
        case "train":
            metrics = list(conf.metrics.val)
        case "test":
            metrics = list(conf.metrics.test)
        case _:
            metrics = []

    if "hlv" not in batch:
        batch["hlv"] = {}
    if "sphereratio" in metrics:
        batch["hlv"] |= {
            f"sphereratio_{k}": v for k, v in sphereratio(batch).items()
        }
    if "cyratio" in metrics:
        batch["hlv"] |= {f"cyratio_{k}": v for k, v in cyratio(batch).items()}
    if "fpc" in metrics:
        batch["hlv"] |= {f"fpc_{k}": v for k, v in fpc_from_batch(batch).items()}
    if "showershape" in metrics:
        batch["hlv"] |= {
            f"showershape_{k}": v for k, v in analyze_layers(batch).items()
        }
    if "response" in metrics:
        batch["hlv"] |= {
            "nhits_n": batch.n_pointsv,
            "nhits_n_by_E": batch.n_pointsv / batch.y[:, 0],
        }
        batch["hlv"]["response"] = response(batch)

    return batch
