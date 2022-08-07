from pathlib import Path
from re import A

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset
from offline_rl.model import FCNN
from offline_rl.model.diffusion import DiffusionFCNN, DiffusionResNet
from offline_rl.utils.dataset.diffusion import get_diffusion_params, xor
from offline_rl.utils.logger import Logger

if True:

    model_nb = 11
    model_name = "{:04d}".format(model_nb)

    save_path = Path("results/models/diffusion_" + str(model_name))
    save_path.mkdir(exist_ok=True, parents=True)

    # create model
    model = DiffusionResNet()
    # model = DiffusionFCNN()
    model.save(save_path / "model_init")
    optimizer = th.optim.Adam(model.parameters(), lr=2e-4)

    # create dataset
    dataset_path = Path("results/dataset/dataset_0")
    ds = load_dataset(dataset_path)
    # ds.all_states

    logger_path = Path("results/loggers/logger_" + str(model_name))
    logger = Logger(
        logger_path,
        "res_net " + str(model_name),
        "cnn, 3 layers, 32 channels",
        save=False,
    )

    x0s, _ = ds.generate_rollout_reward_dataset()
    x0s = x0s[:, 0:3, 0:3, :]
    x0s = np.transpose(x0s, axes=(0, 3, 2, 1))
    # x0s = x0s[:8]
    x0s = np.stack([x0s[0], x0s[1], x0s[3], x0s[4]], axis=0)
    # x0s = np.stack([x0s[0], x0s[1]], axis=0)
    # x0s = np.stack([x0s[0], x0s[4]], axis=0)
    x0s = np.concatenate([x0s], axis=0)
    x0s = th.Tensor(x0s)

    all_losses = []

    # for some epoch
    n_epoch = 10000 * 10
    steps = n_epoch // 10

    # batch_size = 5
    for epoch in range(n_epoch):
        loss = model.loss(x0s)

        full_loss = loss  # + reg

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        # save model
        if (epoch + 1) % steps == 0:
            model.save(save_path / ("model_" + str(epoch // steps)))

        # report statistics
        all_losses.append(full_loss.item())
        logger.log("full_loss", full_loss.item())
        if epoch % 1 == 0:
            print(f"loss: {full_loss.item():>7f}  [{epoch:>5d}/{n_epoch:>5d}]")

    plt.plot(all_losses)
    plt.show()


if False:
    save_path = Path("results/models/diffusion_0")
    save_path.mkdir(exist_ok=True, parents=True)

    # create model
    model = FCNN([3, 32, 32, 2], last_activation=False)
    model.save(save_path / "model_init")
    optimizer = th.optim.Adam(model.parameters(), lr=1e-4)

    # create dataset
    x0s = xor(1000)
    print(x0s.shape)
    # exit()

    T = 1000
    alphas, alphas_bar, betas, betas_tilde = get_diffusion_params(T)

    all_losses = []

    eps = 1e-4

    # for some epoch
    n_epoch = 10000
    steps = n_epoch // 10
    for epoch in range(n_epoch):
        # train model

        t = np.random.randint(T)
        epsilon = np.random.normal(size=x0s.shape)

        xts = np.sqrt(alphas_bar[t]) * x0s + np.sqrt(1 - alphas_bar[t]) * epsilon
        targets = epsilon

        inps = np.concatenate([xts, np.ones((xts.shape[0], 1)) * t / T], axis=1)

        inps = th.Tensor(inps)
        targets = th.Tensor(targets)

        loss = th.mean(th.square(model(inps) - targets))

        full_loss = loss  # + reg

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        # save model
        if (epoch + 1) % steps == 0:
            model.save(save_path / ("model_" + str(epoch // steps)))

        # report statistics
        all_losses.append(full_loss.item())
        if epoch % 1 == 0:
            print(f"loss: {full_loss.item():>7f}  [{epoch:>5d}/{n_epoch:>5d}]")

    plt.plot(all_losses)
    plt.show()
