import importlib
from typing import Callable, List

from fgsim.config import conf

# Import the specified processing sequence
sel_seq = importlib.import_module(f"fgsim.io.{conf.loader.qf_seq_name}")

Event = sel_seq.Event
Batch = sel_seq.Batch
GenOutput = sel_seq.GenOutput


class Postprocessor:
    def __init__(self):
        self.unstack_GenOutput: Callable = sel_seq.unstack_GenOutput
        self.postprocess_event: Callable = sel_seq.postprocess_event
        self.aggregate_to_batch: Callable[
            List[Event], Batch
        ] = sel_seq.aggregate_to_batch
        # self.pppool = None Pool(
        #     conf.loader.num_workers_postprocess,
        # )

    def __call__(self, batch: GenOutput) -> Batch:
        event_list = [x for x in self.unstack_GenOutput(batch)]
        pp_events: List[Event] = [
            self.postprocess_event(event) for event in event_list
        ]
        pp_batch = self.aggregate_to_batch(pp_events)
        return pp_batch

    def __del__(self):
        pass
        # self.pppool.close()
