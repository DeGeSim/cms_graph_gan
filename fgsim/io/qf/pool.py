from collections.abc import Iterable
from multiprocessing.queues import Empty

from torch import multiprocessing as mp

from ...utils.count_iterations import CountIterations
from ...utils.logger import logger
from .step_base import StepBase
from .terminate_queue import TerminateQueue


class PoolStep(StepBase):
    """Class for simple processing steps pooled over multiple workes.
    Each incoming object is processed by a multiple subprocesses
    per worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        nworkers: int,
        **kwargs,
    ):
        # Spawn only one process with deamonize false that can spawn the Pool
        kwargs["deamonize"] = False

        # Make sure the contructor of the base class only initializes
        # one process that manages the pool
        self.n_pool_workers = nworkers
        kwargs["nworkers"] = 1
        super().__init__(*args, **kwargs)

    def start(self):
        for p in self.processes:
            p.daemon = self.deamonize
            p.start()

    def stop(self):
        for p in self.processes:
            p.join(60)
            if p.exitcode is None:
                raise RuntimeError
        self._close_queues()

    def process_status(self):
        return (
            sum([p.is_alive() for p in self.processes]) * self.n_pool_workers,
            self.n_pool_workers,
        )

    def propagete_error(self, element, error=Exception):
        workermsg = f"""{self.workername} failed on element of type {type(element)}."""
        self.error_queue.put((workermsg, element, error))

    def _worker(self):
        self.set_workername()
        logger.info(
            f"{self.workername} pool  initalizing with {self.n_pool_workers} subprocesses"
        )
        self.pool = mp.Pool(self.n_pool_workers)

        while True:
            if self.shutdown_event.is_set():
                break
            try:
                wkin = self.inq.get(block=True, timeout=0.005)
            except Empty:
                continue
            logger.debug(
                f"""{self.workername} working on {id(wkin)} of type {type(wkin)} from queue {id(self.inq)}."""
            )
            wkin = self._clone_tensors(wkin)
            # If the process gets a TerminateQueue object,
            # it terminates the pool and and puts the terminal element in
            # in the outgoing queue.
            if isinstance(wkin, TerminateQueue):
                logger.info(f"{self.workername} terminating")
                self.outq.put(TerminateQueue())
                self.pool.terminate()
                break
            else:
                assert isinstance(wkin, Iterable)
                logger.debug(
                    f"{self.workername} got element"
                    + f" {id(wkin)} of element type {type(wkin)}."
                )
                wkin_iter = CountIterations(wkin)

                try:
                    # wkout_async_res = self.pool.map_async(
                    #     self.workerfn, wkin_iter,
                    # )
                    # while True:
                    #     if wkout_async_res.ready():
                    #         wkout = wkout_async_res.get()
                    #         break
                    #     elif self.shutdown_event.is_set():
                    #         break
                    #     wkout_async_res.wait(1)
                    # if self.shutdown_event.is_set():
                    #     break
                    wkout = self.pool.map(self.workerfn, wkin_iter)

                except Exception as error:
                    self.propagete_error(wkin, error)
                    break
                finally:
                    if wkin_iter.count > 200:
                        logger.warn(
                            f"""\
Giving large iterables ({wkin_iter.count})\
to a worker can lead to crashes.
Lower the number here if you see an error like \
'RuntimeError: unable to mmap x bytes from file </torch_x>:
Cannot allocate memory'"""
                        )
                logger.debug(
                    f"""\
{self.workername} push pool output list {id(wkout)} with \
element type {type(wkin)} into output queue {id(self.outq)}."""
                )
                self.outq.put(wkout)
                del wkin_iter
                del wkin
        self.pool.close()
        self.pool.terminate()
        self._close_queues()
