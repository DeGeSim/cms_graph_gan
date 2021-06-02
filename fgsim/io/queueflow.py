# %%
import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import time
from collections.abc import Iterable

import numpy as np

from ..utils.logger import logger


class TerminateQueue:
    pass


terminate_queue = TerminateQueue()

print_lock = multiprocessing.Lock()


def info_with_lock(*args, **kwargs):
    print_lock.acquire()
    logger.info(*args, **kwargs)
    print_lock.release()


def debug_with_lock(*args, **kwargs):
    print_lock.acquire()
    logger.debug(*args, **kwargs)
    print_lock.release()


def print_with_lock(*args, **kwargs):
    print_lock.acquire()
    print(*args, **kwargs)
    print_lock.release()


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
        debug_with_lock(f"Queuing {i} elements complete")
        self.outq.put(terminate_queue)

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


class Step_Base:
    """Base class"""

    def __init__(
        self,
        workerfn=None,
        nworkers=1,
        inq=multiprocessing.Queue(),
        outq=multiprocessing.Queue(),
        deamonize=True,
        name=None,
    ):
        self.inq = inq
        self.outq = outq
        self.name = type(self) if name is None else name
        self.workerfn = workerfn
        self.nworkers = nworkers
        self.deamonize = deamonize
        self.processes = [
            multiprocessing.Process(target=self._worker) for _ in range(nworkers)
        ]

    def start(self):
        for p in self.processes:
            p.daemon = self.deamonize
            p.start()

    def process_status(self):
        return (sum([p.is_alive() for p in self.processes]), self.nworkers)

    def _worker(self):
        raise NotImplementedError


class Process_Step(Step_Base):
    """Class for simple processing steps.
    Each incoming object is processed by a single worker into a single outgoing element."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.running_processes_counter = multiprocessing.Value(
            multiprocessing.sharedctypes.ctypes.c_uint
        )
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value = 0
        self.first_to_finish_lock = multiprocessing.RLock()

    def _worker(self):
        name = multiprocessing.current_process().name
        info_with_lock(f"{self.name} {name} start working")
        with self.running_processes_counter.get_lock():
            self.running_processes_counter.value += 1
        while True:
            debug_with_lock(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            debug_with_lock(
                f"{self.name} worker {name} working on {id(wkin)} of type {type(wkin)}."
            )
            # If the process gets the terminate_queue object,
            # wait for the others and put it in the next queue
            if isinstance(wkin, TerminateQueue):
                info_with_lock(f"{self.name} Worker {name} terminating")
                # Tell the other workers, that you are finished
                with self.running_processes_counter.get_lock():
                    self.running_processes_counter.value -= 1

                # Put the terminal element back in the input queue
                self.inq.put(terminate_queue)

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
                    self.outq.put(terminate_queue)

                break
            else:
                wkout = self.workerfn(wkin)
                debug_with_lock(
                    f"{self.name} worker {name} push single output of type {type(wkout)} into output queue {id(self.outq)}."
                )
                self.outq.put(wkout)


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
        info_with_lock(
            f"{self.name} pool ({name}) initalizing with {self.nworkers} subprocesses"
        )
        self.pool = multiprocessing.Pool(self.nworkers)
        while True:
            info_with_lock(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()

            # If the process gets the terminate_queue object,
            # it terminates the pool and and puts the terminal element in
            # in the outgoing queue.
            if isinstance(wkin, TerminateQueue):
                info_with_lock(f"{self.name} Worker {name} terminating")
                self.outq.put(terminate_queue)
                self.pool.terminate()
                break
            else:
                assert isinstance(wkin, Iterable)
                info_with_lock(
                    f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
                )
                wkout = self.pool.map(self.workerfn, wkin)
                info_with_lock(
                    f"{self.name} push pool output list {id(wkout)}  with element type {type(wkin)} into output queue {id(self.outq)}."
                )
                self.outq.put(wkout)


class Unpack_Step(Step_Base):
    """A single process takes an iterable from the incoming queue and
    puts the elements one-by-one in the outgoing queue."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = "Unpack"
        super().__init__(*args, **kwargs)

    def _worker(self):
        name = multiprocessing.current_process().name
        info_with_lock(f"{self.name} {name} start working")
        while True:
            debug_with_lock(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            if isinstance(wkin, TerminateQueue):
                debug_with_lock(
                    f"{self.name} push terminal element of type {type(wkin)} into output queue {id(self.outq)}."
                )
                self.outq.put(terminate_queue)
                debug_with_lock(f"{self.name} Worker {name} terminating")
                break
            else:
                assert isinstance(wkin, Iterable)
                info_with_lock(
                    f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
                )
                for e in wkin:
                    debug_with_lock(
                        f"{self.name} push element of type {type(wkin)} into output queue {id(self.outq)}."
                    )
                    self.outq.put(e)


class Pack_Step(Step_Base):
    """Takes an iterable from the incoming queue and
    puts the elements one-by-one in the outgoing queue."""

    def __init__(
        self,
        nelements,
        *args,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = f"Pack({nelements})"
        super().__init__(*args, **kwargs)
        self.nelements = nelements

    def _worker(self):
        name = multiprocessing.current_process().name
        info_with_lock(f"{self.name} {name} start working")
        collected_elements = []
        while True:
            debug_with_lock(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            debug_with_lock(
                f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
            )
            if isinstance(wkin, TerminateQueue):
                if len(collected_elements) > 0:
                    print_with_lock(
                        f"{self.name} terminal element of type {type(wkin)} into output queue {id(self.outq)}."
                    )
                    self.outq.put(collected_elements)
                info_with_lock(f"{self.name} Worker {name} terminating")
                self.outq.put(terminate_queue)
                break
            else:
                debug_with_lock(f"{self.name} storing {id(wkin)} of type {type(wkin)}.")
                collected_elements.append(wkin)
                if len(collected_elements) == self.nelements:
                    debug_with_lock(
                        f"{self.name} push list of type {type(wkin)} into output queue {id(self.outq)}."
                    )
                    self.outq.put(collected_elements)
                    collected_elements = []


class Sequence:
    def __init__(self, *seq):
        self._iterable_is_set = False
        self.seq = [Input_Step(), *seq, Output_Step()]

        self.processes = [p for p in self.seq if isinstance(p, Step_Base)]
        # Chain the processes and queues
        i = 1
        while i < len(self.seq):
            if isinstance(self.seq[i], (Step_Base, Output_Step)):
                # Connect output of the previous process step to
                #  the input of the current process step
                new_queue = multiprocessing.Queue()
                self.seq[i - 1].outq = new_queue
                self.seq[i].inq = new_queue
            elif isinstance(self.seq[i], multiprocessing.queues.Queue):
                # Make sure we are not connecting queues with each other
                assert isinstance(self.seq[i + 1], (Step_Base, Output_Step))
                assert isinstance(self.seq[i - 1], (Step_Base, Input_Step))
                # Connect output of the previous process step to the current pipe
                self.seq[i - 1].outq = self.seq[i]
                self.seq[i + 1].inq = self.seq[i]
                # skip the next connection
                i += 1
            else:
                raise Exception
            i += 1

        # Make the sequence of queues accessable
        self.queues = []
        for e in self.seq:
            if isinstance(e, (multiprocessing.queues.Queue, Output_Step)):
                continue
            else:
                self.queues.append(e.outq)

    def __iter__(self):
        assert (
            self._iterable_is_set
        ), "Must call with iterable: MP_Pipe_Sequence(...)(iterable)"
        return self.seq[-1]

    def __call__(self, iterable):
        self.seq[0].iterable = iterable
        self._iterable_is_set = True
        for e in self.seq:
            if not isinstance(e, multiprocessing.queues.Queue):
                e.start()
        return self

    def queue_status(self):
        return [
            (q.qsize(), q._maxsize if q._maxsize != 2147483647 else "inf")
            for q in self.queues
        ]

    def process_status(self):
        return [p.process_status() for p in self.processes]

    def flowstatus(self):
        qs = self.queue_status()
        ps = self.process_status()
        outstr = "Current Status of Processes and Queues:\n"
        for i in range(len(qs) + len(ps)):
            if i % 2 == 0:
                outstr = (
                    outstr + f"Queue:\t\t{qs[int(i/2)][0]}/{qs[int(i/2)][1]}\n"
                )
            else:
                pscur = ps[i // 2]
                pcur = self.processes[i // 2]
                outstr = outstr + f"Processes:\t{pscur[0]}/{pscur[1]}\t"
                outstr = (
                    outstr + f"({pcur.name if pcur.name is not None else type(pcur)})\n"
                )
        return outstr


# Usage example
# This only run as a standalone file because the function given to pool must be pickleable,
# and if this is called from another file, the defined function has no connection to the top level module and therefore cannot be pickled.s
# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function

# def sleep_times_two(inp):
#     name = multiprocessing.current_process().name
#     print(f"!!!sleep_times_two {name} got input {inp} start sleep")
#     # time.sleep(1)
#     print(f"!!!sleep_times_two {name} finish sleep")
#     return inp * 2


# def minus_one(inp):
#     return inp - 1


# def printqueue(inp):
#     print_with_lock(inp)
#     return inp


# print("Starting queue example")

# process_seq = Sequence(
#     Pack_Step(8),
#     Process_Step(printqueue, 1, name="printfunction1"),
#     Pool_Step(sleep_times_two, nworkers=5, name="sleep_times_two"),
#     Process_Step(printqueue, 1, name="printfunction2"),
#     Unpack_Step(),
#     Process_Step(minus_one, nworkers=5, name="minus_one"),
# )

# res = process_seq(np.random.randint(0, 50, 19))

# oldflowstatus = ""
# for i in range(60):
#     newflowstatus = res.flowstatus()
#     if newflowstatus != oldflowstatus:
#         print_with_lock(newflowstatus)
#         oldflowstatus = newflowstatus
#     else:
#         print_with_lock("+", end="")
#     time.sleep(00.1)

# print("foo")

# for i, e in enumerate(res):
#     print_with_lock(f"({i})Final Output {e}")
#     print_with_lock(res.flowstatus())

# print_with_lock("Done Iterating")
# %%
