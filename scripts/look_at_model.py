from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.model.mdn.simple_model import SimpleMDN
from offline_rl.utils.dataset.synthetic import one_dim_wrapper, step, torch_wrapper
from offline_rl.utils.torch import to_numpy

xs_train, ys_train = step(10000)

# create model
model = SimpleMDN(1, 1)

for i in [9]:
    # for i in range(9, 10):
    save_path = Path("results/models/model_" + str(i))
    model.load(save_path)

    # create dataset
    xs_tensor = th.Tensor(np.linspace(np.min(xs_train), np.max(xs_train))).view((-1, 1))
    pi, sigma, mu = model(xs_tensor)
    a = pi.softmax(dim=-1)
    xs, a, sigma, mu = to_numpy(xs_tensor, a, sigma, mu)

    print(pi.shape)

    plt.figure(figsize=(8, 5))
    for i in range(mu.shape[1]):
        plt.plot(xs[:, 0], mu[:, i, 0], alpha=np.mean(a[:, i]))
        plt.fill_between(
            xs[:, 0],
            mu[:, i, 0] - sigma[:, i],
            mu[:, i, 0] + sigma[:, i],
            alpha=np.mean(a[:, i]),
        )
    plt.plot(xs_train, ys_train, "xk")
    plt.figure(figsize=(8, 5))
    for i in range(mu.shape[1]):
        plt.plot(xs[:, 0], a[:, i])

    plt.show()
    # exit()
