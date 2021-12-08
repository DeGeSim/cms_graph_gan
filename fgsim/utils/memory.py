import gc
import os
import sys
from pprint import pprint
from typing import Optional

import torch

from fgsim.config import device
from fgsim.monitoring.logger import logger


def memReport():

    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            size = obj.view(-1).size()[0]
        else:
            size = sys.getsizeof(obj)
        if size > 1e8:
            if hasattr(obj, "__str__"):
                name = obj.__str__
            elif "__str__" in dir(obj):
                name = obj.__str__()
            if hasattr(obj, "name"):
                name = obj.name
            else:
                name = "Unknown"
            if torch.is_tensor(obj) or type(obj) == dict or len(name) > 20:
                logger.info(f"Type {type(obj)} {size*0.000001}MB")
                pprint(obj)
            else:
                logger.info(f"{name}\t {size*0.000001}MB")


def memGB():
    import psutil

    # logger.debug(psutil.cpu_percent())
    # logger.debug(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB... I think
    # logger.debug("memory GB:", memoryUse)
    return memoryUse


class Gpu_Mem_Monitor:
    def __init__(self):
        self.gpu_mem_res = torch.cuda.memory_reserved(device)
        self.gpu_mem_avail = torch.cuda.memory_allocated(device)

    def print_delta(self, msg: Optional[str] = None):
        gc.collect()
        torch.cuda.empty_cache()
        rp = torch.cuda.memory_reserved(device)
        ap = torch.cuda.memory_allocated(device)

        if msg is None:
            msg = ""
        else:
            msg = msg + " "

        logger.info(
            f"{msg[:25]:>25} GPU Memory Δ: reserved"
            f" {rp-self.gpu_mem_res:+.2E} allocated {ap-self.gpu_mem_avail:+.2E}"
        )
        self.gpu_mem_res = rp
        self.gpu_mem_avail = ap

    def print(self, msg: Optional[str] = None):
        gc.collect()
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        if msg is None:
            msg = ""
        else:
            msg = msg + " "
        logger.info(
            f"{msg[:25]:>25} GPU Memory: reserved {r:.2E} allocated"
            f" {a:.2E} avail {r-a:.2E} total {t:.2E}"
        )


gpu_mem_monitor = Gpu_Mem_Monitor()
