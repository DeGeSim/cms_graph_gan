from torch import multiprocessing

from ...utils.logger import logger
from .terminate_queue import TerminateQueue


class Input_Step:
    """Internal class to read in the iterable into a the first queue"""

    def __init__(self, iterable=(), outq=multiprocessing.Queue()):
        self._iterable = iterable
        self.outq = outq
        self.process = multiprocessing.Process(target=self.load_queue)

    def load_queue(self):
        i = 0
        for e in self._iterable:
            self.outq.put(e)
            i = i + 1
        logger.debug(f"Queuing {i} elements complete")
        self.outq.put(TerminateQueue())

    def start(self):
        self.process.daemon = True
        self.process.start()

    @property
    def iterable(self):
        return self._iterable

    @iterable.setter
    def iterable(self, iterable):
        assert hasattr(iterable, "__iter__")
        self._iterable = iterable


class Output_Step:
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
