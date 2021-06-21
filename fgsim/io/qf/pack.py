from collections.abc import Iterable

import torch
from torch import multiprocessing

from ...utils.logger import logger
from .step_base import Step_Base
from .terminate_queue import TerminateQueue


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
                    f"{self.name} push terminal element of type "
                    + f"{type(wkin)} into output queue {id(self.outq)}."
                )
                self.outq.put(TerminateQueue())
                logger.debug(f"{self.name} Worker {name} terminating")
                break
            else:
                assert isinstance(wkin, Iterable)
                logger.debug(
                    f"{self.name} worker {name} got element "
                    + f"{id(wkin)} of element type {type(wkin)}."
                )
                for e in wkin:
                    logger.debug(
                        f"{self.name} push element of type "
                        + f"{type(wkin)} into output queue {id(self.outq)}."
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
                f"{self.name} worker {name} reading "
                + f"from input queue {id(self.inq)}."
            )
            wkin = self.inq.get()
            logger.debug(
                f"{self.name} worker {name} got element "
                + f"{id(wkin)} of element type {type(wkin)}."
            )
            if isinstance(wkin, torch.Tensor):
                wkin = wkin.clone()
            if isinstance(wkin, TerminateQueue):
                if len(collected_elements) > 0:
                    logger.debug(
                        f"{self.name} terminal element of type"
                        + f" {type(wkin)} into output queue {id(self.outq)}."
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
                        f"{self.name} push list of type {type(collected_elements[-1])}"
                        + f" into output queue {id(self.outq)}."
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
                f"{self.name} worker {name} got element {id(wkin)} "
                + f"of element type {type(wkin)}."
            )
            if isinstance(wkin, TerminateQueue):
                if len(collected_elements) > 0:
                    logger.info(
                        f"{self.name} terminal element of type {type(wkin)} "
                        + f"into output queue {id(self.outq)}."
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
                            f"{self.name} push list of type \
{type(collected_elements[-1])} with {self.nelements} \
elements into output queue {id(self.outq)}."
                        )
                        self.outq.put(collected_elements)
                        collected_elements = []
            del wkin
