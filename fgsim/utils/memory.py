import gc
import os
import pprint
import sys
from typing import Optional

import psutil
import torch

from fgsim.config import device
from fgsim.monitoring.logger import logger


def mem_report():
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
            if torch.is_tensor(obj) or isinstance(obj, dict) or len(name) > 20:
                logger.info(f"Type {type(obj)} {size*0.000001}MB")
                pprint.pprint(obj)
            else:
                logger.info(f"{name}\t {size*0.000001}MB")


def mem_gb():
    # logger.debug(psutil.cpu_percent())
    # logger.debug(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_use = process.memory_info()[0] / 2.0 ** 30  # memory use in GB... I think
    # logger.debug("memory GB:", memoryUse)
    return memory_use


class GpuMemMonitor:
    def __init__(self):
        self.gpu_mem_res = torch.cuda.memory_reserved(device)
        self.gpu_mem_alloc = torch.cuda.memory_allocated(device)
        self.sizes = {}
        self.lastvar: Optional[str] = None
        self.overwrite: bool = True

    def _update_mem(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.gpu_mem_res = torch.cuda.memory_reserved(device)
        self.gpu_mem_alloc = torch.cuda.memory_allocated(device)

    def print(self, msg: Optional[str] = None):
        self._update_mem()
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = self.gpu_mem_res
        allocated = self.gpu_mem_alloc
        if msg is None:
            msg = ""
        else:
            msg = msg + " "
        logger.info(
            f"{msg[:25]:>25} GPU Memory: reserved {reserved:.2E} allocated"
            f" {allocated:.2E} avail {reserved-allocated:.2E} total {total:.2E}"
        )

    def print_delta(self, msg: Optional[str] = None):
        gc.collect()
        torch.cuda.empty_cache()
        cur_reserved = torch.cuda.memory_reserved(device)
        cur_alloc = torch.cuda.memory_allocated(device)

        if msg is None:
            msg = ""
        else:
            msg = msg + " "

        logger.info(
            f"{msg[:15]:>15} GPU Memory Δ: reserved"
            f" {cur_reserved-self.gpu_mem_res:+.2E} allocated"
            f" {cur_alloc-self.gpu_mem_alloc:+.2E}"
        )
        self.gpu_mem_res = cur_reserved
        self.gpu_mem_alloc = cur_alloc

    # Context manager methods
    # with manager("batch"):
    #     batch.to(device)
    # manager.print_recorded()
    def __call__(self, varname: str, overwrite: bool = True):
        assert self.lastvar is None, "Already recording memory footprint."
        self.lastvar = varname
        self.overwrite = overwrite
        self._update_mem()
        return self

    def __enter__(self):
        assert self.lastvar is not None, (
            "Call this with the name of the variable the memory change should be"
            " assigned to."
        )
        return self

    def __exit__(self, *args, **kwargs):
        if self.overwrite or self.lastvar not in self.sizes:
            cur_alloc = torch.cuda.memory_allocated(device)
            self.sizes[self.lastvar] = cur_alloc - self.gpu_mem_alloc
            assert self.sizes[self.lastvar] >= 0, "Size cannot be negative"
        self.lastvar = None

    def print_recorded(self):
        self.sizes = dict(sorted(self.sizes.items(), key=lambda x: x[1]))

        logger.warning(
            pprint.pformat({key: f"{val:.2E}" for key, val in self.sizes.items()})
        )
        sizes_sum = sum(self.sizes.values())
        self._update_mem()
        frac_recorded = sizes_sum / self.gpu_mem_alloc
        logger.warning(
            f"Fraction recorded {frac_recorded*100:.0f}%, missing"
            f" {self.gpu_mem_alloc-frac_recorded:.2E}"
        )


gpu_mem_monitor = GpuMemMonitor()
