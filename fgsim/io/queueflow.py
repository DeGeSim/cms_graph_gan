# %%
import ctypes
import multiprocessing.sharedctypes
# Make it work ()
# import resource
import time
from collections.abc import Iterable

import torch
import torch.multiprocessing
from prettytable import PrettyTable

from ..utils.count_iterations import Count_Iterations
from ..utils.logger import logger

# Two recommendations by
# https://github.com/pytorch/pytorch/issues/973
# 1. (not needed for the moment)
# Traceback (most recent call last):
#   File "$pythonlibpath/threading.py", line 932, in _bootstrap_inner
#     self.run()
#   File "$pythonlibpath/threading.py", line 870, in run
#     self._target(*self._args, **self._kwargs)
#   File "$pythonlibpath/multiprocessing/pool.py", line 576, in _handle_results
#     task = get()
#   File "$pythonlibpath/multiprocessing/connection.py", line 251, in recv
#     return _ForkingPickler.loads(buf.getbuffer())
#   File "$pythonlibpath/site-packages/torch/multiprocessing/reductions.py", line 282, in rebuild_storage_fd
#     fd = df.detach()
#   File "$pythonlibpath/multiprocessing/resource_sharer.py", line 58, in detach
#     return reduction.recv_handle(conn)
#   File "$pythonlibpath/multiprocessing/reduction.py", line 189, in recv_handle
#     return recvfds(s, 1)[0]
#   File "$pythonlibpath/multiprocessing/reduction.py", line 164, in recvfds
#     raise RuntimeError('received %d items of ancdata' %
# RuntimeError: received 0 items of ancdata
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# resource.setrlimit(resource.RLIMIT_NOFILE, (2000 , rlimit[1]))
# 2.
# Without the following option it crashes with
#   File ".../multiprocessing/reduction.py", line 164, in recvfds
#     raise RuntimeError('received %d items of ancdata' %
# RuntimeError: received 0 items of ancdata
torch.multiprocessing.set_sharing_strategy("file_system")

# Reworked according to the recommendations in
# https://pytorch.org/docs/stable/multiprocessing.html

# It works event though multiprocessing with these input is not
#  torch.multiprocessing but just the standard multiprocessing.


class TerminateQueue:
    pass


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

    def stop(self):
        for p in self.processes:
            p.join(10)
            p.terminate()

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
                except:
                    logger.error(
                        f"{self.name} worker {name} failer on element of type of type {type(wkin)}.\n\n{wkin}"
                    )
                    raise Exception
                    exit(1)

                logger.debug(
                    f"{self.name} worker {name} push single output of type {type(wkout)} into output queue {id(self.outq)}."
                )
                self.outq.put(wkout)
                del wkin


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
        logger.info(
            f"{self.name} pool ({name}) initalizing with {self.nworkers} subprocesses"
        )
        self.pool = multiprocessing.Pool(self.nworkers)
        while True:
            logger.debug(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            if isinstance(wkin, torch.Tensor):
                wkin = wkin.clone()
            # If the process gets a TerminateQueue object,
            # it terminates the pool and and puts the terminal element in
            # in the outgoing queue.
            if isinstance(wkin, TerminateQueue):
                logger.info(f"{self.name} Worker {name} terminating")
                self.outq.put(TerminateQueue())
                self.pool.terminate()
                break
            else:
                assert isinstance(wkin, Iterable)
                logger.debug(
                    f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
                )
                wkin_iter = Count_Iterations(wkin)

                try:
                    wkout = self.pool.map(self.workerfn, wkin_iter)
                except:
                    logger.error(
                        f"{self.name} worker {name} failer on element of type of type {type(wkin)}.\n\n{wkin}"
                    )
                    exit(1)
                finally:
                    if wkin_iter.count > 100:
                        logger.warn(
                            f"Giving large iterables ({wkin_iter.count}) to a Pool worker can lead to crashes."
                            + "Lower the number here if you see an error like "
                            + "'RuntimeError: unable to mmap x bytes from file </torch_x>: Cannot allocate memory'"
                        )
                logger.debug(
                    f"{self.name} push pool output list {id(wkout)}  with element type {type(wkin)} into output queue {id(self.outq)}."
                )
                self.outq.put(wkout)
                del wkin_iter
                del wkin


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
        logger.info(f"{self.name} {name} start working")
        while True:
            logger.debug(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            if isinstance(wkin, TerminateQueue):
                logger.debug(
                    f"{self.name} push terminal element of type {type(wkin)} into output queue {id(self.outq)}."
                )
                self.outq.put(TerminateQueue())
                logger.debug(f"{self.name} Worker {name} terminating")
                break
            else:
                assert isinstance(wkin, Iterable)
                logger.debug(
                    f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
                )
                for e in wkin:
                    logger.debug(
                        f"{self.name} push element of type {type(wkin)} into output queue {id(self.outq)}."
                    )
                    if isinstance(e, torch.Tensor):
                        e = e.clone()
                    self.outq.put(e)
                del wkin


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
        logger.info(f"{self.name} {name} start working")
        collected_elements = []
        while True:
            logger.debug(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            logger.debug(
                f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
            )
            if isinstance(wkin, torch.Tensor):
                wkin = wkin.clone()
            if isinstance(wkin, TerminateQueue):
                if len(collected_elements) > 0:
                    logger.debug(
                        f"{self.name} terminal element of type {type(wkin)} into output queue {id(self.outq)}."
                    )
                    self.outq.put(collected_elements)
                logger.info(f"{self.name} Worker {name} terminating")
                self.outq.put(TerminateQueue())
                break
            else:
                logger.debug(f"{self.name} storing {id(wkin)} of type {type(wkin)}.")
                collected_elements.append(wkin)

                if len(collected_elements) == self.nelements:
                    logger.debug(
                        f"{self.name} push list of type {type(collected_elements[-1])} into output queue {id(self.outq)}."
                    )
                    self.outq.put(collected_elements)
                    collected_elements = []
            del wkin


class Repack_Step(Step_Base):
    """Takes an iterable from the incoming queue,
    collects n elements and packs them as a list in the outgoing queue."""

    def __init__(
        self,
        nelements,
        *args,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = f"Repack({nelements})"
        super().__init__(*args, **kwargs)
        self.nelements = nelements

    def _worker(self):
        name = multiprocessing.current_process().name
        logger.info(f"{self.name} {name} start working")
        collected_elements = []
        while True:
            logger.debug(
                f"{self.name} worker {name} reading from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            logger.debug(
                f"{self.name} worker {name} got element {id(wkin)} of element type {type(wkin)}."
            )
            if isinstance(wkin, TerminateQueue):
                if len(collected_elements) > 0:
                    logger.info(
                        f"{self.name} terminal element of type {type(wkin)} into output queue {id(self.outq)}."
                    )
                    self.outq.put(collected_elements)
                logger.info(f"{self.name} Worker {name} terminating")
                self.outq.put(TerminateQueue())
                break
            else:
                assert hasattr(wkin, "__iter__")
                logger.debug(
                    f"{self.name} storing {id(wkin)} of type {type(wkin)} "
                    + f"(len {len(wkin) if hasattr(wkin,'__len__') else '?'})."
                )
                for e in wkin:
                    if isinstance(e, torch.Tensor):
                        e_cloned = e.clone()
                        collected_elements.append(e_cloned)
                    else:
                        collected_elements.append(e)
                    if len(collected_elements) == self.nelements:
                        logger.debug(
                            f"{self.name} push list of type {type(collected_elements[-1])} with {self.nelements} elements into output queue {id(self.outq)}."
                        )
                        self.outq.put(collected_elements)
                        collected_elements = []
            del wkin


class Sequence:
    def __init__(self, *seq):
        self._iterable_is_set = False
        self.seq = [Input_Step(), *seq, Output_Step()]

        self.steps = [p for p in self.seq if isinstance(p, Step_Base)]
        # Chain the processes and queues
        i = 1
        while i < len(self.seq):
            if isinstance(self.seq[i], (Step_Base, Output_Step)):
                # Connect output of the previous process step to
                #  the input of the current process step
                new_queue = multiprocessing.Queue(1)
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

    def stop(self):
        for q in self.queues:
            while not q.empty():
                try:
                    q.get(False)
                except multiprocessing.queues.Empty:
                    continue
            q.put(TerminateQueue())
        for step in self.steps:
            step.stop()

    def queue_status(self):
        return [
            (q.qsize(), q._maxsize if q._maxsize != 2147483647 else "inf")
            for q in self.queues
        ]

    def process_status(self):
        return [p.process_status() for p in self.steps]

    def process_names(self):
        return [
            ",".join([p.name.split("-")[1] for p in step.processes])
            for step in self.steps
        ]

    def flowstatus(self):
        qs = self.queue_status()
        ps = self.process_status()
        pn = self.process_names()
        table = PrettyTable()
        table.title = "Current Status of Processes and Queues"
        table.field_names = ["Type", "Saturation", "Name", "Process names"]
        for i in range(len(qs) + len(ps)):
            if i % 2 == 0:
                table.add_row(["Queue", f"{qs[int(i/2)][0]}/{qs[int(i/2)][1]}", "", ""])
            else:
                pscur = ps[i // 2]
                pncur = pn[i // 2]
                pcur = self.steps[i // 2]
                table.add_row(
                    [
                        "Process",
                        f"{pscur[0]}/{pscur[1]}",
                        pcur.name if pcur.name is not None else type(pcur),
                        pncur,
                    ]
                )
        return table


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
