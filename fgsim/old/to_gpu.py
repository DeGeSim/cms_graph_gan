import torch
from torch import multiprocessing as mp
from .terminate_queue import TerminateQueue

# def to_gpu_fct(workername, inq, outq, device, TerminateQueue, logger):
#     name = mp.current_process().name
#     logger.info(f"{workername} {name} start working")
#     while True:
#         logger.debug(
#             f"{workername} worker {name} reading from input queue {id(inq)}."
#         )
#         logger.debug("Getting item")
#         inp = inq.get()
#         if isinstance(inp, TerminateQueue):
#             outq.put(TerminateQueue())
#             break
#         outp = inp.to(device)
#         outq.put(outp)
#         logger.debug("Done")


def to_gpu_fct(workername, inq, outq, device, TerminateQueue, logger):
    name = mp.current_process().name
    logger.info(f"{workername} {name} start working")
    while True:
        logger.debug(f"{workername} worker {name} reading from input queue {id(inq)}.")
        wkin = inq.get()

        if isinstance(wkin, torch.Tensor):
            wkin = wkin.clone()

        # If the process gets the terminate_queue object,
        # wait for the others and put it in the next queue
        if isinstance(wkin, TerminateQueue):
            logger.info(f"{workername} Worker {name} terminating")
            outq.put(TerminateQueue())
            break
        else:
            try:
                wkout = wkin.to(device)
            except RuntimeError:
                logger.error(
                    f"{workername} worker {name} failed "
                    + f"on element of type {type(wkin)}.\n\n{wkin}"
                )
                try:
                    print(wkin)
                finally:
                    raise RuntimeError

            logger.debug(
                f"{workername} worker {name} push single "
                + f"output of type {type(wkout)} into output queue {id(outq)}."
            )
            outq.put(wkout)
            del wkin


if mp.current_process().name == "MainProcess":
    context = mp.get_context("spawn")
    mp.set_sharing_strategy("file_system")

    from ...config import device
    from ...utils.logger import logger
    from .step_base import StepBase

    class ToGPUStep(StepBase):
        """Process step for the sole purpopose of moving stuff to the GPU."""

        def __init__(self):
            # Because of the spawn method, the inq and outq cannot be initialized here
            # and must be initalized after the sequence overwrites inq and outq
            self.processes = []
            self.name = "to_gpu"
            self.nworkers = 1
            self.deamonize = True

        def start(self):
            # enable restarting
            exitcodes = [process.exitcode for process in self.processes]
            assert all([code == 0 for code in exitcodes]) or all(
                [code is None for code in exitcodes]
            )
            if all([code == 0 for code in exitcodes]):
                # Restart the processes
                self.processes = [
                    context.Process(
                        target=to_gpu_fct,
                        args=(
                            self.name,
                            self.inq,
                            self.outq,
                            device,
                            TerminateQueue,
                            logger,
                        ),
                    )
                ]

            for p in self.processes:
                p.daemon = self.deamonize
                p.start()
