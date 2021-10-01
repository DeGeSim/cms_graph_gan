from tblib import pickling_support
from torch import multiprocessing as mp

from .pack import PackStep, RepackStep, UnpackStep
from .pool import PoolStep
from .process_step import ProcessStep
from .sequence import Sequence

pickling_support.install()


# Two recommendations by
# https://github.com/pytorch/pytorch/issues/973
# 1. (not needed for the moment)
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2000 , rlimit[1]))
# 2.
# Without the following option it crashes with
#   File ".../multiprocessing/reduction.py", line 164, in recvfds
#     raise RuntimeError('received %d items of ancdata' %
# RuntimeError: received 0 items of ancdata


# Make it work ()
# mp.set_sharing_strategy("file_descriptor")
mp.set_sharing_strategy("file_system")

# Reworked according to the recommendations in
# https://pytorch.org/docs/stable/multiprocessing.html

# It works event though multiprocessing with these input is not
#  torch.multiprocessing but just the standard multiprocessing.


__all__ = ["pack", "process_step", "sequence", "pool"]


# Usage example
# This only run as a standalone file because the function given to pool must be pickleable,
# and if this is called from another file, the defined function has no connection
#  to the top level module and therefore cannot be pickled.s
# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function

# def sleep_times_two(inp):
#     name = multiprocessing.current_process().name
#     print(f"!!!sleep_times_two {name} got input {inp} start sleep")
#     # time.sleep(1)
#     print(f"!!!sleep_times_two {name} finish sleep")
#     return inp * 2


# def minus_one(inp):
#     return inp - 1


# def printqueue(inp):
#     print_with_lock(inp)
#     return inp

# process_seq = Sequence(
#     Pack_Step(8),
#     Process_Step(printqueue, 1, name="printfunction1"),
#     Pool_Step(sleep_times_two, nworkers=5, name="sleep_times_two"),
#     Process_Step(printqueue, 1, name="printfunction2"),
#     Unpack_Step(),
#     Process_Step(minus_one, nworkers=5, name="minus_one"),
# )

# res = process_seq(np.random.randint(0, 50, 19))

# oldflowstatus = ""
# for i in range(60):
#     newflowstatus = res.flowstatus()
#     if newflowstatus != oldflowstatus:
#         print_with_lock(newflowstatus)
#         oldflowstatus = newflowstatus
#     else:
#         print_with_lock("+", end="")
#     time.sleep(00.1)

# print("foo")

# for i, e in enumerate(res):
#     print_with_lock(f"({i})Final Output {e}")
#     print_with_lock(res.flowstatus())

# print_with_lock("Done Iterating")
# %%
