"""Contain the training procedure, access point from __main__"""

import signal
import sys
import traceback

from queueflow import Sequence

import wandb
from fgsim.config import device
from fgsim.ml import Holder, Trainer
from fgsim.monitoring import logger
from fgsim.utils.senderror import send_error


def training_procedure() -> None:
    holder = Holder(device)
    trainer = Trainer(holder)
    term_handler = SigTermHander(holder, trainer.loader.qfseq)
    # Regular run
    if sys.gettrace() is not None:
        trainer.training_loop()
    # Debugger is running
    else:
        exitcode = 0
        try:
            trainer.training_loop()
        except Exception:
            exitcode = 1
            tb = traceback.format_exc()
            send_error(tb)
            logger.error(tb)
        finally:
            logger.error("Error detected, stopping qfseq.")
            if trainer.loader.qfseq.started:
                trainer.loader.qfseq.stop()
            exit(exitcode)
    del term_handler


class SigTermHander:
    def __init__(self, holder: Holder, qfseq: Sequence) -> None:
        self.qfseq = qfseq
        self.holder = holder
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, _signo, _stack_frame):
        print("SIGTERM detected, stopping qfseq")
        #  self.holder.save_checkpoint()
        self.qfseq.stop()
        self.holder.save_checkpoint()
        wandb.mark_preempting()

        exit()
