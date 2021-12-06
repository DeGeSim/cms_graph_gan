import importlib
from typing import Callable, List

from torch.multiprocessing import Pool

from fgsim.config import conf, device

# Import the specified processing sequence
sel_seq = importlib.import_module(f"fgsim.io.{conf.loader.qf_seq_name}")

Event = sel_seq.Event
Batch = sel_seq.Batch
GenOutput = sel_seq.GenOutput


class Postprocessor:
    def __init__(self):
        self.unbatch: Callable[Batch, List[Event]] = sel_seq.unbatch
        self.postprocess_event: Callable[
            GenOutput, Event
        ] = sel_seq.postprocess_event
        self.aggregate_to_batch: Callable[
            List[Event], Batch
        ] = sel_seq.aggregate_to_batch
        self.pppool = Pool(
            conf.loader.num_workers_postprocess,
        )

    def __call__(self, batch: GenOutput) -> Batch:
        event_list = self.unbatch(batch).to("cpu")
        pp_events = self.pppool.map(
            self.postprocess_event,
            event_list,
        )
        del event_list
        pp_batch = self.aggregate_to_batch(pp_events)
        del pp_events
        return pp_batch.to(device)

    def __del__(self):
        self.pppool.close()
