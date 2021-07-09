# Queueflow notes:



https://github.com/pytorch/pytorch/issues/13246

Quote: mprostock

I don't know how multiprocessing really works under the hood of pytorch, but we have extensively discussed this "Memory Leak" issue (which probably isn't a memory leak!) on the fast.ai forums (https://forums.fast.ai/t/runtimeerror-dataloader-worker-is-killed-by-signal/31277/55?u=marcmuc). Preliminary findings which hopefully add some insight here (if this does NOT apply, please comment!):

Python Multiprocessing: There is no way of storing arbitrary python objects (even simple lists) in shared memory in Python without triggering copy-on-write behaviour due to the addition of refcounts, everytime something reads from these objects. The refcounts are added memory-page by memory-page, which is why the consumption grows slowly. The processes (workers) will end up having all/most of the memory copied over bit by bit, which is why we get the memory overflow problem. Best description of this behavior is here (SO).

Possible Solution:
Using Multiprocessing like now: in order for python multiprocessing to work without these refcount effects, the objects have to be made “compatible with” and wrapped in multiprocessing.Array before the process pool is created and workers are forked. This supposedly ensures, that the memory will really be shared and no copy-on-write happens. This explains how to do it for numpy arrays and this explains the reasoning behind it again. Don’t get confused by some false statements even by the authors of these good answers stating that copy-on-write makes all of this unnecessary, which is not true. One comment also points to this:

    “Just to note, on Python fork() actually means copy on access (because just accessing the object will change its ref-count).”

I am not familiar with the torch.multiprocessing drop-in replacement that I understand pytorch uses, but I would assume it will also not be able to remove the core python refcount issue.
