import numpy as np
import torch as th


def to_torch(numpy_generator):
    for args in numpy_generator:
        yield (th.Tensor(x) for x in args)


def batchify(*args, batch_size):
    data_size = args[0].shape[0]
    n_batches = data_size // batch_size
    ids = np.arange(data_size)
    scrambeled_ids = np.random.permutation(ids)
    for batch_id in range(n_batches):
        lid = batch_id * batch_size
        hid = lid + batch_size
        ids = scrambeled_ids[lid:hid]
        yield (x[ids] for x in args)


def to_np(th_generator):
    for args in th_generator:
        yield (x.detach().cpu().numpy() for x in args)
