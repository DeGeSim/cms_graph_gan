import threading

import torch
import torch_geometric
from torch import multiprocessing as mp
from types import GeneratorType

class StepBase:
    """Base class"""

    def __init__(
        self,
        workerfn=None,
        nworkers=1,
        deamonize=True,
        name=None,
    ):
        self.name = type(self) if name is None else name
        self.workerfn = workerfn
        self.nworkers = nworkers
        self.deamonize = deamonize
        self.processes = [
            mp.Process(target=self._worker, daemon=self.deamonize)
            for _ in range(self.nworkers)
        ]

    def connect_to_sequence(
        self, input_queue, output_queue, error_queue, shutdown_event
    ):
        self.inq = input_queue
        self.outq = output_queue
        self.error_queue = error_queue
        self.shutdown_event = shutdown_event

    def _close_queues(self):
        self.outq.close()
        self.outq.join_thread()
        self.inq.close()
        self.inq.join_thread()
        self.error_queue.close()
        self.error_queue.join_thread()

    def _clone_tensors(self, wkin):
        if isinstance(wkin, list):
            wkin = [self._clone_tensors(e) for e in wkin]
        elif isinstance(wkin, GeneratorType):
            return (self._clone_tensors(e) for e in wkin )
        elif isinstance(wkin, torch_geometric.data.batch.Data):

            def clone_or_copy(e):
                if torch.is_tensor(e):
                    return e.clone()
                elif isinstance(e, list):
                    return [clone_or_copy(ee) for ee in e]
                elif e is None:
                    return None
                else:
                    raise ValueError

            wkin = torch_geometric.data.Batch().from_dict(
                {k: clone_or_copy(v) for k, v in wkin.to_dict().items()}
            )
        elif isinstance(wkin, torch.Tensor):
            wkin = wkin.clone()
        return wkin

    def set_workername(self):
        self.workername = self.name + "-" + mp.current_process().name.split("-")[1]
        mp.current_process().name = self.workername
        threading.current_thread().name = "MainThread-" + self.workername

    def start(self):
        for p in self.processes:
            p.start()

    def stop(self):
        for p in self.processes:
            p.terminate()

    def process_status(self):
        return (sum([p.is_alive() for p in self.processes]), self.nworkers)

    def _worker(self):
        raise NotImplementedError
