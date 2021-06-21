from multiprocessing import sharedctypes
import time

import torch
from torch import multiprocessing

from ...utils.logger import logger
from .step_base import Step_Base
from .terminate_queue import TerminateQueue


class Process_Step(Step_Base):
    """Class for simple processing steps.
    Each incoming object is processed by a
    single worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.running_processes_counter = multiprocessing.Value(
            sharedctypes.ctypes.c_uint
        )
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value = 0
        self.first_to_finish_lock = multiprocessing.RLock()

    def _worker(self):
        name = multiprocessing.current_process().name
        logger.info(f"{self.name} {name} start working")
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value += 1
        while True:
            logger.debug(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            if isinstance(wkin, torch.Tensor):
                wkin = wkin.clone()
            logger.debug(
                f"{self.name} worker {name} working on {id(wkin)} of type {type(wkin)}."
            )
            # If the process gets the terminate_queue object,
            # wait for the others and put it in the next queue
            if isinstance(wkin, TerminateQueue):
                logger.info(f"{self.name} Worker {name} terminating")
                # Tell the other workers, that you are finished
                with self.running_processes_counter.get_lock():
                    self.running_processes_counter.value -= 1

                # Put the terminal element back in the input queue
                self.inq.put(TerminateQueue())

                # Make the first worker to reach the terminal element
                # aquires the lock and waits for the other processes
                # processes to finish and  reduce the number of running processes to 0
                # then it moves the terminal object from the incoming queue to the
                # outgoing one and exits.
                if self.first_to_finish_lock.acquire(block=False):
                    while True:
                        with self.running_processes_counter.get_lock():
                            if self.running_processes_counter.value == 0:
                                break
                        time.sleep(0.01)
                    # Get the remaining the terminal element from the input queue
                    self.inq.get()
                    self.outq.put(TerminateQueue())

                break
            else:
                try:
                    wkout = self.workerfn(wkin)
                except Exception:
                    logger.error(
                        f"{self.name} worker {name} failed "
                        + f"on element of type of type {type(wkin)}.\n\n{wkin}"
                    )
                    raise Exception

                logger.debug(
                    f"{self.name} worker {name} push single "
                    + f"output of type {type(wkout)} into output queue {id(self.outq)}."
                )
                self.outq.put(wkout)
                del wkin
