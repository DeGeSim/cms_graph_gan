import importlib
from typing import Callable, List

from fgsim.config import conf

# Import the specified processing sequence
sel_seq = importlib.import_module(f"fgsim.io.{conf.loader.qf_seq_name}")

Event = sel_seq.Event
Batch = sel_seq.Batch
# GenOutput = sel_seq.GenOutput


class Postprocessor:
    def __init__(self):
        self.postprocess_event: Callable = sel_seq.postprocess_event

    def __call__(self, batch: Batch) -> Batch:
        event_list = [x for x in batch.split()]
        pp_events: List[Event] = [
            self.postprocess_event(event) for event in event_list
        ]
        batch = Batch.from_event_list(*pp_events)
        return batch

    def __del__(self):
        pass
        # self.pppool.close()
