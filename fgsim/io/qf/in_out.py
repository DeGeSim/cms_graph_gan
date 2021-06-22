from torch import multiprocessing

from ...utils.logger import logger
from .terminate_queue import TerminateQueue


class InputStep:
    """Internal class to read in the iterable into a the first queue"""

    def __init__(self, outq=multiprocessing.Queue()):
        self.outq = outq

    def queue_iterable(self, iterable_object):
        assert hasattr(iterable_object, "__iter__")
        i = 0
        for element in iterable_object:
            self.outq.put(element)
            i = i + 1
        logger.debug(f"Queuing {i} elements complete")
        self.outq.put(TerminateQueue())


class OutputStep:
    # Ag generator reads from the queue
    def __init__(self, inq=multiprocessing.Queue()):
        self.inq = inq

    def start(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        out = self.inq.get()
        if isinstance(out, TerminateQueue):
            raise StopIteration
        else:
            return out
