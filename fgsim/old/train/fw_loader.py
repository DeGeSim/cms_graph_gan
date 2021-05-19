import numpy as np

from ..config import conf
from ..utils.logger import logger


def costumLoader(ds):
    logger.debug("Start of epoch.")
    idxs = np.arange(len(ds))
    np.random.shuffle(idxs)
    batch_size = conf.model["batch_size"]
    batched = np.array(
        [
            idxs[i * batch_size : (i + 1) * batch_size]
            for i in range(len(ds) // batch_size + (1 if len(ds) % batch_size else 0))
        ],
        dtype=object,
    )
    for batch in batched:
        yield ds[batch]


# def costumLoader(ds):
#     logger.debug("Start of epoch.")
#     idxs = np.arange(len(ds))
#     np.random.shuffle(idxs)
#     begin = time.time()
#     batch_size = conf.model["batch_size"]
#     batched = np.array(
#         [
#             idxs[i * batch_size : (i + 1) * batch_size]
#             for i in range(len(ds) // batch_size + (1 if len(ds) % batch_size else 0))
#         ],
#         dtype=object,
#     )
#     iotime = time.time()
#     logger.debug(f"Loading the batches took {iotime-begin}.")
#     for batch in batched:
#         logger.debug("Batch Start")
#         batch_to_transform = [ds[e] for e in batch]
#         logger.debug("Batch Loaded, starting transform.")
#         foo = []
#         for args in zip(batch_to_transform, batch, range(len(batch_to_transform))):
#             res = transform(args)
#             foo.append(res)
#         finishtime = time.time()
#         logger.debug(f"Batch Complete after running transform for {finishtime -iotime}")
#         yield foo

# # p=Pool(10)
# # Define and keep a Pool to load the Datafrom the dataset
# def costumLoader(ds):
#     logger.debug(f"Start of epoch.")
#     idxs = np.arange(len(ds))
#     np.random.shuffle(idxs)
#     begin = time.time()
#     batch_size = conf.model["batch_size"]
#     batched = np.array(
#         [
#             idxs[i * batch_size : (i + 1) * batch_size]
#             for i in range(len(ds) // batch_size + (1 if len(ds) % batch_size else 0))
#         ],
#         dtype=object,
#     )
#     iotime = time.time()
#     logger.debug(f"Loading the batches took {iotime-begin}.")
#     for batch in batched:
#         logger.debug("Batch Start")
#         batch_to_transform = [ds[e] for e in batch]
#         logger.debug("Batch Loaded, starting transform.")
#         # with Pool(10) as p:
#         #     foo = p.map(
#         #         transform, zip(batch_to_transform,batch, range(len(batch_to_transform)))
#         #     )
#         foo = []
#         for args in zip(batch_to_transform, batch, range(len(batch_to_transform))):
#             res = transform(args)
#             foo.append(res)
#         finishtime = time.time()
#         logger.debug(f"Batch Complete after running transform for {finishtime -iotime}")
#         yield foo


# sample = dataset[44302]
# res = transform((sample, 1, 1))
# train_loader = costumLoader(dataset)

# p=Pool(10)
# Define and keep a Pool to load the Datafrom the dataset
