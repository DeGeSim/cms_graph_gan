from torch import multiprocessing as mp

from ...utils.logger import logger
from .terminate_queue import TerminateQueue


class InputStep:
    """Internal class to read in the iterable into a the first queue"""

    def __init__(self, outq=mp.Queue()):
        self.outq = outq
        self.name = "input step"

    def queue_iterable(self, iterable_object):
        assert hasattr(iterable_object, "__iter__")
        i = 0
        for element in iterable_object:
            self.safe_put(self.outq, element)
            i = i + 1
        logger.debug(f"Queuing {i} elements complete")
        self.safe_put(self.outq, TerminateQueue())


class OutputStep:
    """Internal generator class to returning the outputs from the last queue."""

    def __init__(self, inq=mp.Queue()):
        self.inq = inq
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
