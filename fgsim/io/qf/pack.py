from collections.abc import Iterable
from multiprocessing.queues import Empty

from ...utils.logger import logger
from .step_base import StepBase
from .terminate_queue import TerminateQueue


class UnpackStep(StepBase):
    """A single process takes an iterable from the incoming queue and
    puts the elements one-by-one in the outgoing queue."""

    def __init__(self):
        super().__init__(name="Unpack")

    def _terminate(self):
        logger.debug(
            f"{self.workername} push terminal element into output queue {id(self.outq)}."
        )
        self.outq.put(TerminateQueue())
        self._close_queues()
        logger.info(f"{self.workername} terminating")

    def _worker(self):
        self.set_workername()
        logger.info(f"{self.workername} start working")
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
            if isinstance(wkin, TerminateQueue):
                break
            else:
                if not isinstance(wkin, Iterable):
                    errormsg = (
                        f"{self.workername} cannot iterate over "
                        f" {id(wkin)} of element type {type(wkin)}."
                    )
                    self.error_queue.put((errormsg, wkin, ValueError))
                    break
                logger.debug(
                    f"{self.workername} got element "
                    + f"{id(wkin)} of element type {type(wkin)}."
                )
                for element in wkin:
                    logger.debug(
                        f"{self.workername} push element of type "
                        + f"{type(wkin)} into output queue {id(self.outq)}."
                    )
                    if hasattr(element, "clone"):
                        element = self._clone_tensors(element)
                    self.outq.put(element)
                del wkin
        self._terminate()


class PackStep(StepBase):
    """Takes an iterable from the incoming queue and
    puts the elements one-by-one in the outgoing queue."""

    def __init__(
        self,
        nelements,
    ):
        super().__init__(name=f"Pack({nelements})")
        self.nelements = nelements
        self.collected_elements = []

    def _terminate(self):
        if len(self.collected_elements) > 0:
            logger.debug(
                f"{self.workername} terminal element of type {type(self.collected_elements[0])}"
                + f" into output queue {id(self.outq)}."
            )
            self.outq.put(self.collected_elements)
        logger.info(f"{self.workername} terminating")
        self.outq.put(TerminateQueue())
        self._close_queues()

    def _worker(self):
        self.set_workername()
        logger.info(f"{self.workername} start working")
        while True:
            if self.shutdown_event.is_set():
                break
            try:
                wkin = self.inq.get(block=True, timeout=0.005)
            except Empty:
                continue
            logger.debug(
                f"{self.workername} working on {id(wkin)} of type {type(wkin)} from queue {id(self.inq)}."
            )
            wkin = self._clone_tensors(wkin)
            if isinstance(wkin, TerminateQueue):
                break
            else:
                logger.debug(
                    f"{self.workername} storing {id(wkin)} of type {type(wkin)}."
                )
                self.collected_elements.append(wkin)

                if len(self.collected_elements) == self.nelements:
                    logger.debug(
                        f"{self.workername} push list of type {type(self.collected_elements[-1])}"
                        + f" into output queue {id(self.outq)}."
                    )
                    self.outq.put(self.collected_elements)
                    self.collected_elements = []
            del wkin
        self._terminate()


class RepackStep(StepBase):
    """Takes an iterable from the incoming queue,
    collects n elements and packs them as a list in the outgoing queue."""

    def __init__(self, nelements):
        super().__init__(name=f"Repack({nelements})")
        self.nelements = nelements
        self.collected_elements = []

    def _terminate(self):
        if len(self.collected_elements) > 0:
            logger.debug(
                f"{self.workername} terminal element of type {type(self.collected_elements[0])}"
                + f" into output queue {id(self.outq)}."
            )
            self.outq.put(self.collected_elements)
        logger.info(f"{self.workername} terminating")
        self.outq.put(TerminateQueue())
        self._close_queues()

    def _worker(self):
        self.set_workername()
        logger.info(f"{self.workername} start working")
        while True:
            if self.shutdown_event.is_set():
                break
            try:
                wkin = self.inq.get(block=True, timeout=0.005)
            except Empty:
                continue
            logger.debug(
                f"{self.workername} working on {id(wkin)} of type {type(wkin)} from queue {id(self.inq)}."
            )
            if isinstance(wkin, TerminateQueue):
                break
            else:
                if not isinstance(wkin, Iterable):
                    errormsg = (
                        f"{self.workername} cannot iterate over "
                        f" {id(wkin)} of element type {type(wkin)}."
                    )
                    self.error_queue.put((errormsg, wkin, ValueError))
                    break

                logger.debug(
                    f"{self.workername} storing {id(wkin)} of type {type(wkin)} "
                    + f"(len {len(wkin) if hasattr(wkin,'__len__') else '?'})."
                )
                for element in wkin:
                    e_cloned = self._clone_tensors(element)
                    self.collected_elements.append(e_cloned)
                    if len(self.collected_elements) == self.nelements:
                        logger.debug(
                            f"{self.workername} push list of type {type(self.collected_elements[-1])} "
                            f"with {self.nelements} elements into"
                            f" output queue {id(self.outq)}."
                        )
                        self.outq.put(self.collected_elements)
                        self.collected_elements = []
            del wkin
        self._terminate()
