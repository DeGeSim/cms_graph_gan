# %%
import multiprocessing
import numpy as np
import time


class TerminateQueue:
    pass


terminate_queue = TerminateQueue()

print_lock = multiprocessing.Lock()

def print_with_lock(s):
    print_lock.acquire()
    print(s)
    print_lock.release()

class MP_Pipe_Input:
    def __init__(self, iterable=(), outq=multiprocessing.Queue()):
        self._iterable = iterable
        self.outq = outq
        self.process = multiprocessing.Process(target=self.load_queue)

    def load_queue(self):
        i=0
        for e in self._iterable:
            self.outq.put(e)
            i=i+1
        print_with_lock(f"Queuing {i} elements complete")
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


class MP_Pipe_Process_Step:
    def __init__(
        self,
        workerfn,
        nworkers,
        inq=multiprocessing.Queue(),
        outq=multiprocessing.Queue(),
        deamonize = True,
    ):
        self.workerfn = workerfn
        self.nworkers = nworkers
        self.inq = inq
        self.outq = outq
        self.deamonize=deamonize
        self.processes = [
            multiprocessing.Process(target=self._worker) for _ in range(nworkers)
        ]

    def start(self):
        for p in self.processes:
            p.daemon = self.deamonize
            p.start()

    def join(self):
        for p in self.processes:
            p.join()

    def _worker(self):
        name = multiprocessing.current_process().name
        print_with_lock(f"{name} start working")
        while True:
            # print_with_lock(f"{name} trying to read from queue {self.inq}")
            wkin = self.inq.get()
            # print_with_lock(f"{name} got Element of type {type(wkin)}")
            # If the process gets the terminate_queue object, wait for the others and put it in the next queue
            if isinstance(wkin, TerminateQueue):
                print_with_lock(f"Worker {name} terminating")
                self.outq.put(terminate_queue)
                self.inq.put(terminate_queue)
                # print_with_lock(f"Queue status q1 {self.inq.qsize()} q2 {self.outq.qsize()}")
                break
            else:
                wkout = self.workerfn(wkin)
                # print_with_lock(f"Worker {name} finished task")
                self.outq.put(wkout)


class MP_Pipe_Output:
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


class MP_Pipe_Sequence:
    def __init__(self, *seq):
        self._iterable_is_set = False
        self.seq = [MP_Pipe_Input(), *seq, MP_Pipe_Output()]
        i = 1
        while i < len(self.seq):
            if isinstance(self.seq[i], (MP_Pipe_Process_Step, MP_Pipe_Output)):
                # Connect output of the previous process step to
                #  the input of the current process step
                self.seq[i - 1].outq = self.seq[i].inq
            elif isinstance(self.seq[i], multiprocessing.queues.Queue):
                # Make sure we are not connecting queues with each other
                assert isinstance(
                    self.seq[i + 1], (MP_Pipe_Process_Step, MP_Pipe_Output)
                )
                assert isinstance(
                    self.seq[i - 1], (MP_Pipe_Process_Step, MP_Pipe_Input)
                )
                # Connect output of the previous process step to the current pipe
                self.seq[i - 1].outq = self.seq[i]
                self.seq[i + 1].inq = self.seq[i]
                # skip the next connection
                i += 1
            else:
                raise Exception
            i += 1


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


# %%
# Usage example 
# def fct(inp):
#     time.sleep(1 + np.random.rand())
#     return inp * 2

# process_seq = MP_Pipe_Sequence(
#     multiprocessing.Queue(2),
#     MP_Pipe_Process_Step(workerfn=fct, nworkers=2),
#     multiprocessing.Queue(1),
# )

# for e in process_seq(list(range(5))):
#     print_with_lock(f"Final Output {e}")
# print_with_lock("Done Iterating")