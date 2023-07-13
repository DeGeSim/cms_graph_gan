from torch_geometric.data import Batch

from fgsim.config import conf

from .convcoord import batch_to_Exyz
from .pca import fpc_from_batch
from .shower import analyze_layers, response, sphereratio
from .voxelize import sum_dublicate_hits


def postprocess(batch: Batch) -> Batch:
    batch = sum_dublicate_hits(batch)
    batch = batch_to_Exyz(batch)
    metrics: list[str] = conf.training.val.metrics
    if "sphereratio" in metrics:
        batch["sphereratio"] = sphereratio(batch)
    if "fpc" in metrics:
        batch["fpc"] = fpc_from_batch(batch)
    if "showershape" in metrics:
        batch["showershape"] = analyze_layers(batch)
    if "response" in metrics:
        batch["response"] = response(batch)

    return batch
