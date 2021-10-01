from collections.abc import Iterable
from multiprocessing.queues import Empty

from fgsim.utils.logger import logger

from .step_base import StepBase
from .terminate_queue import TerminateQueue


class UnpackStep(StepBase):
    """A single process takes an iterable from the incoming queue and
    puts the elements one-by-one in the outgoing queue."""

    def __init__(self):
        super().__init__(name="Unpack")

    def __handle_terminal(self):
        logger.debug(
            f"""\
{self.workername} push terminal element into output queue {id(self.outq)}."""
        )
        self.safe_put(self.outq, TerminateQueue())

    def _worker(self):
        self.set_workername()
        logger.info(f"{self.workername} start working")
        while True:
            if self.shutdown_event.is_set():
                break
            try:
                wkin = self.inq.get(block=True, timeout=0.05)
                wkin = self._clone_tensors(wkin)
            except Empty:
                continue
            logger.debug(
                f"""\
{self.workername} working type {type(wkin)} from queue {id(self.inq)}."""
            )
            if isinstance(wkin, TerminateQueue):
                self.__handle_terminal()
                continue

            if not isinstance(wkin, Iterable):
                errormsg = f"""\
{self.workername} cannot iterate over element type {type(wkin)}."""
                self.error_queue.put((errormsg, wkin, ValueError))
                break
            logger.debug(
                f"{self.workername} got element of element type {type(wkin)}."
            )
            for element in wkin:
                logger.debug(
                    f"""\
{self.workername} push element of type {type(wkin)} into output queue."""
                )
                self.safe_put(self.outq, element)
            del wkin
        self._close_queues()
        logger.info(f"{self.workername} terminating")


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

    def __handle_terminal(self):
        if len(self.collected_elements) > 0:
            logger.debug(
                f"""\
{self.workername} put remainder of size {len(self.collected_elements)} into output queue."""
            )
            self.safe_put(self.outq, self.collected_elements)
        logger.debug(
            f"""\
{self.workername} terminal element into output queue {id(self.outq)}."""
        )
        self.safe_put(self.outq, TerminateQueue())

    def _worker(self):
        self.set_workername()
        logger.info(f"{self.workername} start working")
        while True:
            if self.shutdown_event.is_set():
                break
            try:
                wkin = self.inq.get(block=True, timeout=0.05)
                wkin = self._clone_tensors(wkin)
            except Empty:
                continue
            logger.debug(
                f"""\
{self.workername} working on type {type(wkin)} from queue {id(self.inq)}."""
            )

            if isinstance(wkin, TerminateQueue):
                self.__handle_terminal()
                continue

            logger.debug(
                f"""\
{self.workername} storing element of type {type(wkin)}."""
            )
            self.collected_elements.append(wkin)

            if len(self.collected_elements) == self.nelements:
                logger.debug(
                    f"""\
{self.workername} push list of type \
{type(self.collected_elements[-1])} into output queue {id(self.outq)}."""
                )
                self.safe_put(self.outq, self.collected_elements)
                self.collected_elements = []
            del wkin
        self._close_queues()


class RepackStep(StepBase):
    """Takes an iterable from the incoming queue,
    collects n elements and packs them as a list in the outgoing queue."""

    def __init__(self, nelements):
        super().__init__(name=f"Repack({nelements})")
        self.nelements = nelements
        self.collected_elements = []

    def __handle_terminal(self):
        if len(self.collected_elements) > 0:
            logger.debug(
                f"""\
{self.workername} put remainder of size {len(self.collected_elements)} into output queue."""
            )
            self.safe_put(self.outq, self.collected_elements)
        logger.debug(
            f"""\
{self.workername} terminal element into output queue {id(self.outq)}."""
        )
        self.safe_put(self.outq, TerminateQueue())
        logger.warning(
            f"""\
{self.workername} finished with iterable (in {self.count_in}/out {self.count_out})"""
        )
        self.count_in, self.count_out = 0, 0

    def _worker(self):
        self.set_workername()
        logger.info(f"{self.workername} start working")
        while True:
            if self.shutdown_event.is_set():
                break
            try:
                wkin = self.inq.get(block=True, timeout=0.05)
            except Empty:
                continue
            logger.debug(
                f"""
{self.workername} working on type {type(wkin)} from queue {id(self.inq)}."""
            )
            if isinstance(wkin, TerminateQueue):
                self.__handle_terminal()
                continue
            if not isinstance(wkin, Iterable):
                errormsg = f"""\
{self.workername} cannot iterate over element type {type(wkin)}."""
                self.error_queue.put((errormsg, wkin, ValueError))
                break
            self.count_in += 1
            logger.debug(
                f"""\
{self.workername} storing element of type {type(wkin)} \
(len {len(wkin) if hasattr(wkin,'__len__') else '?'})."""
            )
            for element in wkin:
                self.collected_elements.append(element)
                if len(self.collected_elements) == self.nelements:
                    logger.debug(
                        f"""\
{self.workername} push list of type {type(self.collected_elements[-1])} \
with {self.nelements} elements into output queue {id(self.outq)}."""
                    )
                    self.safe_put(self.outq, self.collected_elements)
                    self.collected_elements = []
                    self.count_out += 1
            del wkin
        self._close_queues()
