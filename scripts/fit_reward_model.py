from pathlib import Path
from re import A

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset
from offline_rl.model import FCNN
from offline_rl.model.diffusion.attention import DiffusionAttentionNet
from offline_rl.model.mdn.convnet import MDNConvNet
from offline_rl.utils.logger import Logger

if True:

    model_nb = 0
    model_name = "{:04d}".format(model_nb)

    save_path = Path("results/models/reward_" + str(model_name))
    save_path.mkdir(exist_ok=True, parents=True)

    # create model
    model = MDNConvNet((7, 8, 8), 1)
    model.save(save_path / "model_init")
    optimizer = th.optim.Adam(model.parameters(), lr=2e-4)

    # create dataset
    dataset_path = Path("results/dataset/dataset_0")
    ds = load_dataset(dataset_path)
    all_states, all_rewards = ds.generate_rollout_reward_dataset()
    print(all_states.shape, all_rewards.shape)

    logger_path = Path("results/loggers/reward_" + str(model_name))
    logger = Logger(
        logger_path,
        "mdn conv net " + str(model_name),
        "cnn, 3 ernel size, 16 channels",
        save=True,
    )

    all_states = th.Tensor(all_states)
    all_rewards = th.Tensor(all_rewards)

    all_losses = []

    # for some epoch
    n_epoch = 10000
    steps = n_epoch // 10

    batch_size = 100

    for epoch in range(n_epoch):
        if batch_size is not None:
            batch_idx = np.random.randint(all_states.shape[0], size=(batch_size,))
            train_states = all_states[batch_idx]
            train_rewards = all_rewards[batch_idx]
        else:
            train_states = all_states
            train_rewards = all_rewards

        loss = th.mean(-model.log_density(train_states, train_rewards))

        full_loss = loss

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
