import time
from multiprocessing import sharedctypes
from multiprocessing.queues import Empty

from torch import multiprocessing as mp

from ...utils.logger import logger
from .step_base import StepBase
from .terminate_queue import TerminateQueue


class ProcessStep(StepBase):
    """Class for simple processing steps.
    Each incoming object is processed by a
    single worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.running_processes_counter = mp.Value(sharedctypes.ctypes.c_uint)
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value = 0
        self.first_to_finish_lock = mp.RLock()

    def _terminate(self, workername):
        logger.info(f"{workername}  trying to terminate")

        # Put the terminal element back in the input queue
        self.safe_put(self.inq, TerminateQueue())

        # Make the first worker to reach the terminal element
        # aquires the lock and waits for the other processes
        # processes to finish and  reduce the number of running processes to 0
        # then it moves the terminal object from the incoming queue to the
        # outgoing one and exits.
        if self.first_to_finish_lock.acquire(block=False):
            logger.info(
                f"{workername} first to encounter"
                f" Terminal element, waiting for the other processes."
            )
            while True:
                with self.running_processes_counter.get_lock():
                    if self.running_processes_counter.value == 1:
                        break
                time.sleep(0.01)
            # Get the remaining the terminal element from the input queue
            self.inq.get()
            self.safe_put(self.outq, TerminateQueue())

            logger.info(f"{workername} put terminal element in outq.")
            self.first_to_finish_lock.release()
        self._close_queues()
        logger.warn(
            f"{self.workername} terminating (in {self.count_in}/out {self.count_out})"
        )
        # Tell the other workers, that you are finished
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value -= 1

    # def _debugtarget(self):
    #     return "do_nothing1" in self.workername and (
    #         int(self.workername.split("-")[-1]) > 38
    #     )
    #     if self._debugtarget():
    #         logger.setLevel(10)

    def _worker(self):
        self.set_workername()

        logger.info(f"{self.workername} start working")
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value += 1

        logger.debug(f"{self.workername} reading from input queue {id(self.inq)}.")
        while not self.shutdown_event.is_set():
            try:
                wkin = self.inq.get(block=True, timeout=0.005)
            except Empty:
                continue
            logger.debug(
                f"""\
{self.workername} working on {id(wkin)} of type {type(wkin)} from queue {id(self.inq)}."""
            )
            # If the process gets the terminate_queue object,
            # wait for the others and put it in the next queue
            if isinstance(wkin, TerminateQueue):
                break
            self.count_in += 1
            # We need to overwrite the method of cloning the batches
            # because we have list of tensors as attibutes of the batch.
            # If copy.deepcopy is called on this object
            wkin = self._clone_tensors(wkin)

            try:
                wkout = self.workerfn(wkin)

            # Catch Errors in the worker function
            except Exception as error:
                workermsg = f"""
{self.workername} failed on element of type of type {type(wkin)}.\n\n{wkin}"""
                self.error_queue.put((workermsg, wkin, error))
                break

            logger.debug(
                f"{self.workername} push single "
                + f"output of type {type(wkout)} into output queue {id(self.outq)}."
            )
            self.safe_put(self.outq, wkout)
            self.count_out += 1
            del wkin
        self._terminate(self.workername)
