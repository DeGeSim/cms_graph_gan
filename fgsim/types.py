import importlib

from fgsim.config import conf

sel_seq = importlib.import_module(f"fgsim.loaders.{conf.loader_name}")

# Event = sel_seq.Event
# Batch = sel_seq.Batch
