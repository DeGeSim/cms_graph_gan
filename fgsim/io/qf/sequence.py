import threading
import time
from multiprocessing.queues import Queue as queues_class

from prettytable import PrettyTable
from torch import multiprocessing

from ...config import conf
from ...utils.logger import logger
from .in_out import InputStep, OutputStep
from .step_base import StepBase
from .terminate_queue import TerminateQueue


class Sequence:
    def __init__(self, *seq):
        self.__iterables_queued = False
        self._started = False
        self.__seq = [InputStep(), *seq, OutputStep()]

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
                        new_queue = multiprocessing.Queue()
                    # Standard for all other steps
                    else:
                        new_queue = multiprocessing.Queue(1)
                    self.__seq.insert(i + 1, new_queue)
            i += 1
        for i, elem in enumerate(self.__seq):
            if i % 2 == 0:
                continue
            assert isinstance(elem, queues_class)
        # Connect the queues
        for i in range(len(self.__seq)):
            if isinstance(self.__seq[i], queues_class):
                # Make sure we are not connecting queues with each other
                assert isinstance(self.__seq[i + 1], (StepBase, OutputStep))
                assert isinstance(self.__seq[i - 1], (StepBase, InputStep))
                # Connect output of the previous process step to the current pipe
                self.__seq[i - 1].outq = self.__seq[i]
                self.__seq[i + 1].inq = self.__seq[i]

        # Make the sequence of queues accessable
        self.queues = [q for q in self.__seq if isinstance(q, queues_class)]
        self.steps = [p for p in self.__seq if isinstance(p, StepBase)]

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
            self.__stop()
            for queue in self.queues:
                assert queue.empty()
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
        # Print the status of the queue once in while
        self.status_printer_thread = threading.Thread(
            target=self.printflowstatus, daemon=True
        )
        self.stop_printer_thread = False
        self.status_printer_thread.start()
        return self

    def __stop(self):
        logger.debug("Before Sequence Stop\n" + str(self.flowstatus()))
        for step in self.steps:
            step.stop()
        self.stop_printer_thread = True
        self.status_printer_thread.join()

    def drain_seq(self):
        terminal_pos = -1
        while terminal_pos < len(self.queues) - 1:
            for iqueue in range(terminal_pos + 1, len(self.queues)):
                queue = self.queues[iqueue]
                while not queue.empty():
                    out = queue.get(False)
                    if isinstance(out, TerminateQueue):
                        terminal_pos = iqueue
                        queue.put(TerminateQueue())
                        continue
        self.__stop()
        logger.debug("\n" + str(self.flowstatus()))

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
        oldflowstatus = ""
        sleeptime = 5 if conf.debug else 10
        while not getattr(self, "stop_printer_thread", True):
            newflowstatus = str(self.flowstatus())
            if newflowstatus != oldflowstatus:
                logger.info("\n" + newflowstatus)
                oldflowstatus = newflowstatus
            time.sleep(sleeptime)
