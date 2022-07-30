from pathlib import Path
from re import A

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.dataset import Dataset, load_dataset
from offline_rl.model import ConvMDN
from offline_rl.utils.dataset.helper import batchify, to_torch

model_save_path = Path("results/models/reward_0")
model_save_path.mkdir(exist_ok=True, parents=True)
dataset_path = Path("results/dataset/dataset_0")

# create model
model = ConvMDN((8, 8, 6), 1)
model.save(model_save_path / "model_init")
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)


# create dataset
all_states, all_rewards = load_dataset(dataset_path).generate_rollout_reward_dataset()
print("available size :", all_states.shape[0])
# all_states = all_states[:2000]
# all_rewards = all_rewards[:2000]
print(all_states.shape, all_rewards.shape)

all_losses = []
all_reg = []
all_full_losses = []
mean_losses = []

# for some epoch
n_epoch = 100
steps = n_epoch // 10
for epoch in range(n_epoch):
    # all_rewards = all_rewards + np.random.normal(0, 1, size=all_rewards.shape) * 0.01
    # train model
    cur_loss = []
    for batch_nb, (states, rewards) in enumerate(
        to_torch(batchify(all_states, all_rewards, batch_size=512))
    ):
        loss = th.mean(-model.log_density(states, rewards))
        reg = th.mean(model.standard_dist(states))  # 1e-4
        full_loss = loss + reg * 0

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        # report statistics
        all_losses.append(loss.item())
        all_reg.append(reg.item())
        all_full_losses.append(full_loss.item())
        if epoch % 1 == 0:
            print(
                f"loss: {full_loss.item():>7f} {reg.item():>7f}  [{epoch:>5d}/{n_epoch:>5d}]"
            )
        cur_loss.append(loss.item())
    mean_losses.append(np.mean(cur_loss))

    # save model
    if (epoch + 1) % steps == 0:
        model.save(model_save_path / ("model_" + str(epoch // steps)))

    if (epoch + 1) % 100 == 0:
        plt.figure()
        plt.plot(all_losses)
        plt.title("all_losses")
        plt.figure()
        plt.plot(all_reg)
        plt.title("all_reg")
        plt.figure()
        plt.plot(all_full_losses)
        plt.title("all_full_losses")
        plt.figure()
        plt.plot(mean_losses)
        plt.title("mean_losses")
        plt.show()
