import importlib

from torch_geometric.data import Batch as GraphBatch
from torch_geometric.data import Data as Graph

from fgsim.config import conf

sel_seq = importlib.import_module(f"fgsim.io.{conf.loader.qf_seq_name}")

Event = sel_seq.Event
Batch = sel_seq.Batch
GraphBatch
Graph
