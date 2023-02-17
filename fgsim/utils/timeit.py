import time
from collections import namedtuple

# from fgsim.monitoring import logger


def timeit(func, n=1):
    ResTuple = namedtuple("ResTuple", ["res", "time"])

    def wrapper(*arg, **kw):
        """source: http://www.daniweb.com/code/snippet368.html"""
        t1 = time.time()
        for i in range(n):
            res = func(*arg, **kw)
        t2 = time.time()
        # logger.info(
        #     func.__name__,(t2 - t1) / n
        # )
        return ResTuple(res, (t2 - t1) / n)

    return wrapper
