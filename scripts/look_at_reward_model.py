from pathlib import Path
from re import A
from venv import create

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset, one_hot_to_string
from offline_rl.model.mdn.convnet import MDNConvNet


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
    model_name = "{:04d}".format(model_nb)

    save_path = Path("results/models/reward_" + str(model_name))
    model = MDNConvNet((7, 8, 8), 1)
    model.load(save_path / "model_9")

    states_np = [create_state(1, 1), create_state(-3, -2)]

    # with th.no_grad():
    #     latents = model.generate_ddim(1000).detach().cpu().numpy()

    for state_np in states_np:
        plt.figure()
        state_th = th.Tensor(state_np)
        states_th = state_th.view((1, 7, 8, 8))

        print(one_hot_to_string(state_np))

        with th.no_grad():
            pi, sigma, mu = model(states_th)

        a_np = pi.softmax(dim=-1).detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        print("predicted mean :", np.sum(a_np.flatten() * mu_np.flatten()))
        xs = np.linspace(-0.25, 1.25, 1000)
        su = xs * 0
        for a, s, m in zip(a_np.flatten(), sigma_np.flatten(), mu_np.flatten()):
            ys = (
                1
                / (np.sqrt(2 * np.pi) * s)
                * np.exp(-0.5 * np.square(xs - m) / np.square(s))
            )
            yp = np.exp(-0.5 * np.square(xs - m) / np.square(s))

            # plt.plot(xs, np.cumsum(ys))
            plt.plot(xs, yp * a)
            su += yp * a
        # plt.plot(xs, np.cumsum(su) * (np.max(xs) - np.min(xs)) / len(xs))
        plt.plot(xs, su)

    # plt.figure()
    # for state_np in states_np:

    #     state_rewards = []
    #     for state, reward in zip(all_states, all_rewards):
    #         if np.all(state == state_np):
    #             state_rewards.append(reward)
    #     plt.hist(
    #         np.array(state_rewards).flatten(), range=(np.min(xs), np.max(xs)), bins=30
    #     )

    plt.show()
