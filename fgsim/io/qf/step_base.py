from torch import multiprocessing


class StepBase:
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
            multiprocessing.Process(target=self._worker) for _ in range(self.nworkers)
        ]

    def start(self):
        # enable restarting
        exitcodes = [process.exitcode for process in self.processes]
        assert all([code == 0 for code in exitcodes]) or all(
            [code is None for code in exitcodes]
        )
        if all([code == 0 for code in exitcodes]):
            # Restart the processes
            self.processes = [
                multiprocessing.Process(target=self._worker)
                for _ in range(self.nworkers)
            ]

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
