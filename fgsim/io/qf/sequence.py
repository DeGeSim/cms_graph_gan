import threading
import time

from prettytable import PrettyTable
from torch import multiprocessing

from ...config import conf
from ...utils.logger import logger
from .in_out import Input_Step, Output_Step
from .step_base import Step_Base
from .terminate_queue import TerminateQueue

from multiprocessing.queues import Queue as queues_class

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
            elif isinstance(self.seq[i], queues_class):
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
            if isinstance(e, (queues_class, Output_Step)):
                continue
            else:
                self.queues.append(e.outq)

        self.status_printer_thread = threading.Thread(
            target=self.printflowstatus, daemon=True
        )
        self.stop_printer_thread = False

    def __iter__(self):
        assert (
            self._iterable_is_set
        ), "Must call with iterable: MP_Pipe_Sequence(...)(iterable)"
        return self.seq[-1]

    def __call__(self, iterable):
        self.seq[0].iterable = iterable
        self._iterable_is_set = True
        for e in self.seq:
            if not isinstance(e, queues_class):
                e.start()
        # Print the status of the queue once in while
        self.status_printer_thread.start()
        return self

    def printflowstatus(self):
        oldflowstatus = ""
        sleeptime = 5 if conf.debug else 10
        while not getattr(self, "stop_printer_thread", True):
            newflowstatus = str(self.flowstatus())
            if newflowstatus != oldflowstatus:
                logger.info("\n" + newflowstatus)
                oldflowstatus = newflowstatus
            time.sleep(sleeptime)

    def stop(self):
        terminal_pos = -1
        while terminal_pos < len(self.queues) - 1:
            for iqueue in range(terminal_pos + 1, len(self.queues)):
                q = self.queues[iqueue]
                while not q.empty():
                    out = q.get()
                    if isinstance(out, TerminateQueue):
                        terminal_pos = iqueue
                        q.put(TerminateQueue())
                        continue
        for step in self.steps:
            step.stop()
        self.stop_printer_thread = True

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
