import multiprocessing
import threading


def thread_or_process():
    thread = threading.current_thread().name
    if thread == "MainThread":
        return f"{multiprocessing.current_process().name}"
    else:
        return f"{thread}:"