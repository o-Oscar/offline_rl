from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset, one_hot_to_string
from offline_rl.model.diffusion.attention import DiffusionAttentionNet
from offline_rl.model.diffusion.convnet import DiffusionConvNet


def board_from_latent(latent):
    max_value = np.max(latent, axis=0, keepdims=True)
    max_value = max_value * np.ones(latent.shape)
    to_return = np.where(latent == max_value, 1, 0)
    return to_return


def count_players(latent):
    return np.sum(latent[:4])


def create_state(x, y):
    state_np = np.zeros((7, 8, 8))
    for i in range(8):
        state_np[4, 0, i] = 1
        state_np[4, -1, i] = 1
        state_np[4, i, 0] = 1
        state_np[4, i, -1] = 1

    state_np[5, -2, -2] = 1
    state_np[0, x, y] = 1

    state_np[6] = 1 - np.max(state_np, axis=0, keepdims=True)

    return state_np


if True:

    model_nb = 0
    model_name = "next_diffusion__{:04d}".format(model_nb)

    save_path = Path("results/models/" + model_name)

    # model = DiffusionConvNet()
    model = DiffusionAttentionNet(7)
    model.load(save_path / "model_9")

    # states_np = [create_state(1, 1) for i in range(100)]
    states_np = [create_state(5, 4) for i in range(100)]
    states_np = np.asarray(states_np)
    # , create_state(-3, -2)]

    with th.no_grad():
        latents = (
            model.generate_ddim(states_np.shape[0], infos=th.Tensor(states_np))
            .detach()
            .cpu()
            .numpy()
        )

    print(latents.shape)

    for latent in latents[:100]:
        print(one_hot_to_string(board_from_latent(latent)))
        print(latent[:, 1, 1])
        print()

    n_higher = len(
        [0 for latent in latents if count_players(board_from_latent(latent)) > 1]
    )
    high_precent = int(np.ceil(100 * n_higher / len(latents)))
    n_lower = len(
        [0 for latent in latents if count_players(board_from_latent(latent)) < 1]
    )
    low_precent = int(np.ceil(100 * n_lower / len(latents)))
    total_precent = int(np.ceil(100 * (n_lower + n_higher) / len(latents)))

    print("too much players : {} ({}%)".format(n_higher, high_precent))
    print("too few players : {} ({}%)".format(n_lower, low_precent))
    print("total : {}%".format(total_precent))
    print()

    # for i in [0, 1]:
    #     plt.figure()
    #     plt.hist(latents[:, 0, i, i], bins=20)
    #     plt.figure()
    #     plt.hist(latents[:, 1, i, i], bins=20)

    #     plt.figure()
    #     xs = latents[:, 0, i, i]
    #     ys = latents[:, 1, i, i]
    #     plt.plot(xs, ys, ".")

    # plt.show()
