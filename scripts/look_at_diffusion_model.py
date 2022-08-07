from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset, one_hot_to_string
from offline_rl.model.diffusion import DiffusionFCNN, DiffusionResNet


def board_from_latent(latent):
    max_value = np.max(latent, axis=0, keepdims=True)
    max_value = max_value * np.ones(latent.shape)
    to_return = np.where(latent == max_value, 1, 0)
    return to_return


if True:

    save_path = Path("results/models/diffusion_0000")
    save_path.mkdir(exist_ok=True, parents=True)

    model = DiffusionResNet()
    model.load(save_path / "model_9")

    with th.no_grad():
        latents = model.generate(1000).detach().cpu().numpy()

    for latent in latents[:10]:
        print(one_hot_to_string(board_from_latent(latent)))
        print(latent[:, 1, 1])
        print()

    for i in [0, 1]:
        plt.figure()
        plt.hist(latents[:, 0, i, i], bins=20)
        plt.figure()
        plt.hist(latents[:, 1, i, i], bins=20)

        plt.figure()
        xs = latents[:, 0, i, i]
        ys = latents[:, 1, i, i]
        plt.plot(xs, ys, ".")

    plt.show()
