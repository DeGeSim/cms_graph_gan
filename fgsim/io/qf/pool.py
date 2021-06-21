from collections.abc import Iterable

import torch
from torch import multiprocessing

from ...utils.count_iterations import Count_Iterations
from ...utils.logger import logger
from .step_base import Step_Base
from .terminate_queue import TerminateQueue


class Pool_Step(Step_Base):
    """Class for simple processing steps pooled over multiple workes.
    Each incoming object is processed by a multiple subprocesses
    per worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Spawn only one process with deamonize false that can spawn the Pool
        kwargs["deamonize"] = False
        assert "nworkers" in kwargs, "Pool_Step needs nworkers argument"

        # Make sure the contructor of the base class only initializes
        # one process that manages the pool
        nworkers = kwargs["nworkers"]
        kwargs["nworkers"] = 1
        super().__init__(*args, **kwargs)
        self.nworkers = nworkers

    def process_status(self):
        return (
            sum([p.is_alive() for p in self.processes]) * self.nworkers,
            self.nworkers,
        )

    def _worker(self):
        name = multiprocessing.current_process().name
        logger.info(
            f"{self.name} pool ({name}) initalizing with {self.nworkers} subprocesses"
        )
        self.pool = multiprocessing.Pool(self.nworkers)
        while True:
            logger.debug(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            if isinstance(wkin, torch.Tensor):
                wkin = wkin.clone()
            # If the process gets a TerminateQueue object,
            # it terminates the pool and and puts the terminal element in
            # in the outgoing queue.
            if isinstance(wkin, TerminateQueue):
                logger.info(f"{self.name} Worker {name} terminating")
                self.outq.put(TerminateQueue())
                self.pool.terminate()
                break
            else:
                assert isinstance(wkin, Iterable)
                logger.debug(
                    f"{self.name} worker {name} got element"
                    + f" {id(wkin)} of element type {type(wkin)}."
                )
                wkin_iter = Count_Iterations(wkin)

                try:
                    wkout = self.pool.map(self.workerfn, wkin_iter)
                except Exception:
                    logger.error(
                        f"{self.name} worker {name} failer on element"
                        + f" of type of type {type(wkin)}.\n\n{wkin}"
                    )
                    exit(1)
                finally:
                    if wkin_iter.count > 200:
                        logger.warn(
                            f"""Giving large iterables ({wkin_iter.count})
to a worker can lead to crashes.
Lower the number here if you see an error like
'RuntimeError: unable to mmap x bytes from file </torch_x>:
Cannot allocate memory'"""
                        )
                logger.debug(
                    f"{self.name} push pool output list {id(wkout)}"
                    + f"  with element type {type(wkin)} into output queue {id(self.outq)}."
                )
                self.outq.put(wkout)
                del wkin_iter
                del wkin
        self.pool.terminate()
