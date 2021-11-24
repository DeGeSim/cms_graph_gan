import os


def push_to_old(path_new, path_old):
    if os.path.isfile(path_new):
        if os.path.isfile(path_old):
            os.remove(path_old)
        os.rename(path_new, path_old)
