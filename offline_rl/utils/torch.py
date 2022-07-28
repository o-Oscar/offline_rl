import numpy as np

import torch as th


def to_numpy(*args: tuple[th.Tensor]):
    return [arg.detach().cpu().numpy() for arg in args]
