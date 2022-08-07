from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset, one_hot_to_string
from offline_rl.model import FCNN
from offline_rl.model.diffusion import DiffusionFCNN, DiffusionResNet
from offline_rl.utils.dataset.diffusion import get_diffusion_params, xor
from offline_rl.utils.dataset.synthetic import one_dim_wrapper, step, torch_wrapper
from offline_rl.utils.torch import to_numpy

# def board_from_latent(latent):
#     latent = np.minimum(np.maximum(latent, 0), 1)
#     last_axis = 1 - np.sum(latent, axis=-1, keepdims=True)
#     full_latent = np.concatenate([latent, last_axis], axis=-1)
#     max_value = np.max(full_latent, axis=-1, keepdims=True)
#     max_value = max_value * np.ones((8, 8, 6))
#     to_return = np.where(latent == max_value, 1, 0)
#     return to_return


def board_from_latent(latent):
    max_value = np.max(latent, axis=0, keepdims=True)
    max_value = max_value * np.ones(latent.shape)
    to_return = np.where(latent == max_value, 1, 0)
    return to_return


if False:
    save_path = Path("results/models/diffusion_0011")
    save_path.mkdir(exist_ok=True, parents=True)

    model = DiffusionResNet()
    model.load(save_path / "model_9")

    save_path = Path("results/dataset/dataset_0")
    ds = load_dataset(save_path)
    # ds.plot_rollout(0)
    # exit()

    original = ds.all_states[0, 0]
    # print(original)
    # exit()
    original = np.stack([original] * 100, axis=0)

    all_res = []
    for i in range(0, 30, 1):
        latents = model.generate_partial(original, i * 3).detach().cpu().numpy()
        # latents = latents / np.sum(latents, axis=3, keepdims=True)
        # cur_res = np.mean(np.max(latents[:, :, :, :4], axis=(1, 2, 3)))
        cur_res = np.mean(latents[:, 1, 1, 0])
        # plt.hist(latents[:, 1, 1, 3])
        all_res.append(cur_res)
        # for latent in latents:
        # print(one_hot_to_string(board_from_latent(latent)))
        # print(np.max(latent[:, :, :4]))
        # print()
        print(i)
    plt.plot(all_res)
    plt.show()


def calc_dist_value(xs):
    higher = np.where(xs > 0.5)
    big = xs[higher]
    lower = np.where(xs < 0.5)
    small = xs[lower]

    mub = np.mean(big)
    sigmab = np.std(big)
    mus = np.mean(small)
    sigmas = np.std(small)

    print(mub, sigmab, mus, sigmas)

    return sigmas**2 + sigmab**2  # + (1 - mub) ** 2 + (mus) ** 2


if False:
    save_path = Path("results/models/diffusion_0011")
    save_path.mkdir(exist_ok=True, parents=True)

    model = DiffusionResNet()
    # model = DiffusionFCNN()
    model.load(save_path / "model_9")

    dataset_path = Path("results/dataset/dataset_0")
    ds = load_dataset(dataset_path)
    x0s, _ = ds.generate_rollout_reward_dataset()
    x0s = x0s[:, 0:3, 0:3, :]
    x0s = np.transpose(x0s, axes=(0, 3, 2, 1))
    x0s = np.stack([x0s[0], x0s[4]], axis=0)
    x0s = np.concatenate([x0s] * 500, axis=0)
    x0s[:, 0, 1, 1] = np.linspace(0, 1, 1000)
    x0s[:, 1, 1, 1] = 1 - np.linspace(0, 1, 1000)
    xts = th.Tensor(x0s)

    for t in range(0, 1000, 100):
        x0s = model.calc_x0(xts, t / 1000, model.alphas_bar[t])
        x0s = x0s.detach().cpu().numpy()

        i = 1
        xs = x0s[:, 0, i, i]
        ys = x0s[:, 1, i, i]
        plt.plot(np.linspace(0, 1, 1000), xs)
    plt.show()

if False:

    save_path = Path("results/models/diffusion_0011")
    save_path.mkdir(exist_ok=True, parents=True)

    model = DiffusionResNet()
    # model = DiffusionFCNN()
    model.load(save_path / "model_9")

    dataset_path = Path("results/dataset/dataset_0")
    ds = load_dataset(dataset_path)
    x0s, _ = ds.generate_rollout_reward_dataset()
    x0s = x0s[:, 0:3, 0:3, :]
    x0s = np.transpose(x0s, axes=(0, 3, 2, 1))
    x0s = np.stack([x0s[0], x0s[4]], axis=0)
    x0s = np.concatenate([x0s] * 500, axis=0)
    x0s = th.Tensor(x0s)
    all_losses = []
    for t in range(0, model.diffusion_steps, 100):
        print(t)
        loss = model.loss_t(x0s, t)
        all_losses.append(loss.item())

    plt.plot(all_losses)
    plt.show()

if True:  # This one

    save_path = Path("results/models/diffusion_0011")
    save_path.mkdir(exist_ok=True, parents=True)

    model = DiffusionResNet()
    # model = DiffusionFCNN()
    model.load(save_path / "model_9")

    print(
        model.alphas[100],
        model.alphas_bar[100],
        model.betas[100],
        model.betas_tilde[100],
    )
    # exit()

    with th.no_grad():
        latents = model.generate(1000).detach().cpu().numpy()

    for i in [0, 1]:
        # latents_th = th.Tensor(latents)
        # x0s = model.calc_x0(latents_th, 0.1, model.alphas_bar[100])
        # x0s = x0s.detach().cpu().numpy()

        plt.figure()
        plt.hist(latents[:, 0, i, i], bins=20)
        plt.figure()
        plt.hist(latents[:, 1, i, i], bins=20)

        plt.figure()
        xs = latents[:, 0, i, i]
        ys = latents[:, 1, i, i]
        plt.plot(xs, ys, ".")
        # xs = x0s[:, 0, i, i]
        # ys = x0s[:, 1, i, i]
        # plt.plot(xs, ys, ".")

        a = calc_dist_value(xs)
        b = calc_dist_value(ys)
        print("dist value :", a + b)
    plt.show()

    for latent in latents[:10]:
        print(one_hot_to_string(board_from_latent(latent)))
        print(latent[:, 1, 1])
        print()


if False:
    save_path = Path("results/models/diffusion_0")
    save_path.mkdir(exist_ok=True, parents=True)

    # create model
    model = FCNN([3, 32, 32, 2], last_activation=False)
    model.load(save_path / "model_9")

    xts = np.random.normal(size=(10000, 2))

    T = 1000
    alphas, alphas_bar, betas, betas_tilde = get_diffusion_params(T)
    sigmas = betas
    sigmas = betas_tilde

    for t in reversed(range(T)):
        z = np.random.normal(size=xts.shape)
        if t == 0:
            z *= 0

        inps = np.concatenate([xts, np.ones(xts.shape[0:1] + (1,)) * t / T], axis=1)
        res = model(th.Tensor(inps)).detach().cpu().numpy()

        epsilon = res

        xts = (
            1
            / np.sqrt(alphas[t])
            * (xts - (1 - alphas[t]) / np.sqrt(1 - alphas_bar[t]) * epsilon)
            + sigmas[t] * z
        )

    plt.plot(xts[:, 0], xts[:, 1], ".")
    plt.show()

if False:
    x = np.linspace(-2, 2, 20)
    xs, ys = np.meshgrid(x, x)

    xss = np.stack([xs, ys], axis=2)
    inps = np.concatenate([xss, np.ones(xss.shape[0:2] + (1,)) * 0.1], axis=2)
    res = model(th.Tensor(inps)).detach().cpu().numpy()
    resx = res[:, :, 0]
    resy = res[:, :, 1]

    plt.quiver(xs, ys, resx, resy)
    plt.show()
