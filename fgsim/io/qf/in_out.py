from multiprocessing.queues import Full

from ...utils.logger import logger
from .terminate_queue import TerminateQueue


class InOutStep:
    def __init__(self):
        pass

    def safe_put(self, queue, element):
        while not self.shutdown_event.is_set():
            try:
                queue.put(element, True, 1)
                break
            except Full:
                continue


class InputStep(InOutStep):
    """Internal class to read in the iterable into a the first queue"""

    def __init__(self):
        self.name = "input step"

    def queue_iterable(self, iterable_object):
        assert hasattr(iterable_object, "__iter__")
        i = 0
        for element in iterable_object:
            self.safe_put(self.outq, element)
            i = i + 1
        logger.debug(f"Queuing {i} elements complete")
        self.safe_put(self.outq, TerminateQueue())

    def connect_to_sequence(self, output_queue, shutdown_event):
        self.outq = output_queue
        self.shutdown_event = shutdown_event


class OutputStep(InOutStep):
    """Internal generator class to returning the outputs from the last queue."""

    def __init__(self):
        self.name = "output step"

    def start(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        out = self.inq.get()
        logger.debug("Sequence output ready.")
        if isinstance(out, TerminateQueue):
            raise StopIteration
        else:
            return out

    def connect_to_sequence(self, input_queue, shutdown_event):
        self.inq = input_queue
        self.shutdown_event = shutdown_event
