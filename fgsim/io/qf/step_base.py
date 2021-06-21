from torch import multiprocessing


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
