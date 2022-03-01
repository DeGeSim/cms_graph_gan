import importlib

from torch_geometric.data import Data as Graph
from torch_geometric.data.batch import DataBatch as GraphBatch

from fgsim.config import conf

sel_seq = importlib.import_module(f"fgsim.loaders.{conf.loader_name}")

Event = sel_seq.Event
Batch = sel_seq.Batch
GraphBatch
Graph
