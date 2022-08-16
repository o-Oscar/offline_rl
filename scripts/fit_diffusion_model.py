from pathlib import Path
from re import A

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset
from offline_rl.model import FCNN
from offline_rl.model.diffusion.attention import DiffusionAttentionNet
from offline_rl.model.diffusion.convnet import DiffusionConvNet
from offline_rl.utils.dataset.diffusion import get_diffusion_params, xor
from offline_rl.utils.logger import Logger

if True:

    model_nb = 6
    model_name = "{:04d}".format(model_nb)

    save_path = Path("results/models/diffusion_" + str(model_name))
    save_path.mkdir(exist_ok=True, parents=True)

    # create model
    # model = DiffusionConvNet()
    model = DiffusionAttentionNet()
    model.save(save_path / "model_init")
    optimizer = th.optim.Adam(model.parameters(), lr=2e-4)

    # create dataset
    dataset_path = Path("results/dataset/test_dataset_6")
    ds = load_dataset(dataset_path)

    logger_path = Path("results/loggers/logger_" + str(model_name))
    logger = Logger(
        logger_path,
        "res_net " + str(model_name),
        "cnn, 10 layers, 16 channels, 3 kernel size",
        save=True,
    )

    # x0s, _ = ds.generate_rollout_reward_dataset()
    x0s = ds.generate_all_states_dataset()
    # x0s = x0s[:, :, 0:3, 0:3]
    # x0s = np.transpose(x0s, axes=(0, 3, 2, 1))
    # x0s = np.stack([x0s[0], x0s[1], x0s[3], x0s[4]], axis=0)
    # ds.plot_rollout(0)
    # exit()
    # x0s = np.concatenate([x0s], axis=0)
    x0s = th.Tensor(x0s)
    print(x0s.shape)

    all_losses = []

    # for some epoch
    n_epoch = 100000
    steps = n_epoch // 10

    batch_size = 4

    for epoch in range(n_epoch):
        batch_idx = np.random.randint(x0s.shape[0], size=(batch_size,))
        train_x0s = x0s[batch_idx]

        loss = model.loss(train_x0s)

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
