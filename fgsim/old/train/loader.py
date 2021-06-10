# https://stackoverflow.com/questions/7323664/python-generator-pre-fetch
import asyncio
import queue
import threading
import time
from multiprocessing import Process, Queue

import numpy as np

ds = np.random.rand(1000)


class MyGen:
    def __init__(self):
        self.idxs = np.arange(len(ds))
        np.random.shuffle(self.idxs)
        self.batch_size = 50
        self.batched = np.array(
            [
                self.idxs[i * self.batch_size : (i + 1) * self.batch_size]
                for i in range(
                    len(ds) // self.batch_size + (1 if len(ds) % self.batch_size else 0)
                )
            ],
            dtype=object,
        )
        self.prefetch = 10

        self.batchidx_inqueue = 0
        self.batchidx_outqueue = 0

        self.inputqueue = asyncio.Queue()
        self.outputqueue = asyncio.Queue()

        self._queuebatches()

        self.tasks = []
        for i in range(3):
            task = asyncio.create_task(self.worker(self.inputqueue, self.outputqueue))
            self.tasks.append(task)

    def _queuebatches(self):
        while self.batchidx_inqueue < self.batchidx_outqueue + self.prefetch:
            if self.batchidx_inqueue == len(self.batched):
                break
            self.inputqueue.put(self.batched[self.batchidx_inqueue])
            self.batchidx_inqueue += 1

    def __iter__(self):
        return self

    async def worker(self, inqueue, outqueue):
        while True:
            time.sleep(1)  # Take a while to process
            batch = await inqueue.get()
            outqueue.put(ds[batch])
            inqueue.task_done()

            print(f"{name} has slept for {sleep_for:.2f} seconds")

    def __del__(self):
        self.stop()

    async def stop(self):
        # Cancel our worker self.tasks.
        for task in self.tasks:
            task.cancel()
        # Wait until all worker self.tasks are cancelled.
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while True:  # Flush the queue
            try:
                self.inputqueue.get(False)
            except queue.Empty:
                break
        while True:  # Flush the queue
            try:
                self.outputqueue.get(False)
            except queue.Empty:
                break

        self.t.join()

    async def next(self):
        # Start a thread to compute the next next.
        # Now deliver the already-queued element
        while True:
            try:
                print("request at", time.time())
                obj = self.outputqueue.get()
                self.outputqueue.task_done()
                self.batchidx_outqueue += 1
                self._queuebatches()
                return obj
            except queue.Empty:
                pass
            time.sleep(0.001)


if __name__ == "__main__":
    f = MyGen()
    for i in range(5):
        print("*********")
        print(f.next())
        print("returned at", time.time())
