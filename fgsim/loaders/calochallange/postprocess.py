from torch_geometric.data import Batch

from fgsim.config import conf

from .convcoord import batch_to_Exyz
from .pca import fpc_from_batch
from .shower import analyze_shower, cone_ratio


def postprocess(batch: Batch) -> Batch:
    batch = batch_to_Exyz(batch)
    metrics: list[str] = conf.loader.validation
    if "coneratio" in metrics:
        batch["coneratio"] = cone_ratio(batch).reshape(-1)
    if "fpc" in metrics:
        batch["fpc"] = fpc_from_batch(batch)
    if "showershape" in metrics:
        batch["showershape"] = analyze_shower(batch)

    return batch
