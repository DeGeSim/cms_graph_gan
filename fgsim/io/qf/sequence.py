import threading
import time
from multiprocessing.queues import Empty
from multiprocessing.queues import Queue as queues_class

from prettytable import PrettyTable
from torch import multiprocessing as mp

from ...config import conf
from ...utils.logger import logger
from .in_out import InputStep, OutputStep
from .step_base import StepBase
from .terminate_queue import TerminateQueue


class Sequence:
    """
    Initialize with a sequence of qf steps (ProcessStep, PoolStep, RePack, Pack).
    Iterables a given into this sequence with the `queue_iterable` function.
    Processes the steps sequentially, iterating over instances of this class will
    yield the outpus as soon as available.

    BUG: If torch_geometric batches are passed from one process to another,
     the subprocesses recieving the batch will crash hard under the following conditions:
     - This class is initialized more than one time.
     - The machine has CUDA not available.
     The for the process reading the batches from the queue the tensors of the batch will be
     printable and accessing single element (eg `batch.x[0][0]+1`) will be possible,
     but `batch.x.clone()` or `batch.x + 1` will lead to a crash.
     This problem may or may not go way by spawning the subprocesses instead of forking them,
     as recommended in the torch.multiprocessing package.
    """

    def __init__(self, *seq):
        self.__iterables_queued = False
        self._started = False
        self.__seq = [InputStep(), *seq, OutputStep()]

        self.shutdown_event = mp.Event()
        self.error_queue = mp.Queue()
        # Chain the processes and queues

        for elem in self.__seq:
            assert isinstance(elem, (queues_class, StepBase, InputStep, OutputStep))
        # Insert the queues in between the steps
        i = 0
        while i < len(self.__seq):
            if isinstance(self.__seq[i], (StepBase, InputStep)):
                if not isinstance(self.__seq[i + 1], queues_class):
                    # Allow the InputQueue to be infinitly big
                    if isinstance(self.__seq[i], InputStep):
                        new_queue = mp.Queue()
                    # Standard for all other steps
                    else:
                        new_queue = mp.Queue(1)
                    self.__seq.insert(i + 1, new_queue)
            i += 1
        for i, elem in enumerate(self.__seq):
            if i % 2 == 0:
                continue
            assert isinstance(elem, queues_class)

        self.queues = [q for q in self.__seq if isinstance(q, queues_class)]
        self.steps = [p for p in self.__seq if isinstance(p, StepBase)]
        # Connect the input:
        self.__seq[0].connect_to_sequence(
            output_queue=self.__seq[1],
            shutdown_event=self.shutdown_event,
        )
        # Connect the output:
        self.__seq[-1].connect_to_sequence(
            input_queue=self.__seq[-2],
            shutdown_event=self.shutdown_event,
        )

        # Set up the processes
        for i, step in enumerate(self.__seq):
            if not isinstance(step, StepBase):
                continue
            step.connect_to_sequence(
                input_queue=self.__seq[i - 1],
                output_queue=self.__seq[i + 1],
                error_queue=self.error_queue,
                shutdown_event=self.shutdown_event,
            )

        # make sure everything is connected properly
        for i in range(len(self.__seq) - 1):
            if isinstance(self.__seq[i], queues_class):
                continue
            assert self.__seq[i].outq is self.__seq[i + 2].inq
            assert self.__seq[i].outq is self.__seq[i + 1]

            # Print the status of the queue once in while
        self.status_printer_thread = threading.Thread(
            target=self.printflowstatus, daemon=True
        )
        # Print the status of the queue once in while
        self.error_queue_thread = threading.Thread(
            target=self.read_error_queue, daemon=True
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.__iterables_queued == 0:
            raise BufferError(
                "No iterable queued: call queueflow.queue_iterable(iterable)"
            )
        if not self._started:
            self.__start()
            self._started = True
        try:
            out = next(self.__seq[-1])
            return out
        except StopIteration:
            logger.debug("Sequence: Stop Iteration encountered.")
            self.__iterables_queued -= 1
            self.shutdown_event.set()
            self._stop()
            for queue in self.queues:
                if queue._closed:
                    continue
                assert queue.empty(), "Some queue is not empty."
            self._started = False
            raise StopIteration

    def queue_iterable(self, iterable):
        self.__seq[0].queue_iterable(iterable)
        self.__iterables_queued += True

    def __start(self):
        logger.debug("Before Sequence Start\n" + str(self.flowstatus()))

        for seq_elem in self.__seq:
            if isinstance(seq_elem, StepBase):
                seq_elem.start()
        for step in self.__seq:
            logger.debug(
                (
                    step.name if hasattr(step, "name") else None,
                    id(step.inq) if hasattr(step, "inq") else None,
                    id(step.outq) if hasattr(step, "outq") else None,
                )
            )

        self.status_printer_thread.start()
        self.error_queue_thread.start()

        return self

    def _stop(self):
        logger.debug("Before Sequence Stop\n" + str(self.flowstatus()))
        self.shutdown_event.set()

        for step in self.steps:
            step.stop()

        self.queues[1].close()
        self.queues[1].join_thread()
        self.queues[-1].close()
        self.queues[-1].join_thread()
        self.error_queue.close()
        self.error_queue.join_thread()
        self.error_queue_thread.join()

    def drain_seq(self):
        logger.warning("qf Sequence is being drained.")
        terminal_pos = -1
        while terminal_pos < len(self.queues) - 1:
            for iqueue in range(terminal_pos + 1, len(self.queues)):
                queue = self.queues[iqueue]
                while not queue._closed and not queue.empty():
                    try:
                        out = queue.get(block=False)
                    except Empty:
                        continue
                    if isinstance(out, TerminateQueue):
                        terminal_pos = iqueue
                        queue.put(TerminateQueue())
                        queue.close()
                        queue.join_thread()
                        continue
        self._stop()
        logger.debug("\n" + str(self.flowstatus()))

    def read_error_queue(self):
        threading.current_thread().setName("readErrorQueue")
        while not self.shutdown_event.is_set() and not self.error_queue._closed:
            try:
                workermsg, wkin, error = self.error_queue.get(
                    block=True, timeout=0.5
                )
                # If there is an error, stop eveything
                self.shutdown_event.set()
                logger.error(workermsg)
                try:
                    logger.error(wkin)
                finally:
                    raise error
            except Empty:
                continue

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
        queues_status = self.queue_status()
        processes_status = self.process_status()
        processes_names = self.process_names()
        table = PrettyTable()
        table.title = "Current Status of Processes and Queues"
        table.field_names = ["Type", "Saturation", "Name", "Process names"]
        for i in range(len(queues_status) + len(processes_status)):
            if i % 2 == 0:
                table.add_row(
                    [
                        "Queue",
                        f"{queues_status[int(i/2)][0]}/{queues_status[int(i/2)][1]}",
                        "",
                        "",
                    ]
                )
            else:
                pscur = processes_status[i // 2]
                pncur = processes_names[i // 2]
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

    def printflowstatus(self):
        threading.current_thread().setName("flowstatusPrinter")
        oldflowstatus = ""
        sleeptime = 5 if conf.debug else 10
        while not self.shutdown_event.is_set():
            newflowstatus = str(self.flowstatus())
            if newflowstatus != oldflowstatus:
                logger.info("\n" + newflowstatus)
                oldflowstatus = newflowstatus
            time.sleep(sleeptime)
