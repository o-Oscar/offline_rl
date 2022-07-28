from pathlib import Path
from re import A

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from offline_rl.model.mdn.simple_model import SimpleMDN
from offline_rl.utils.dataset.synthetic import (
    cross,
    curve,
    linear,
    one_dim_wrapper,
    step,
    torch_wrapper,
)

save_path = Path("results/models/")
save_path.mkdir(exist_ok=True, parents=True)

# create model
model = SimpleMDN(1, 1)
model.save(save_path / "model_init")
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)


# create dataset
xs, ys = one_dim_wrapper(step)(10000)

all_losses = []

# for some epoch
n_epoch = 1000
steps = n_epoch // 10
for epoch in range(n_epoch):
    # train model
    idx = np.random.choice(xs.shape[0], size=1000, replace=False)
    ex, ey = th.Tensor(xs[idx]), th.Tensor(ys[idx])

    loss = th.mean(-model.log_density(ex, ey))
    # loss = th.mean(model.standard_dist(xs))
    reg = th.mean(model.standard_dist(ex)) * 1e-4

    full_loss = loss + reg

    optimizer.zero_grad()
    full_loss.backward()
    optimizer.step()

    # save model
    if (epoch + 1) % steps == 0:
        model.save(save_path / ("model_" + str(epoch // steps)))

    # report statistics
    all_losses.append(full_loss.item())
    if epoch % 1 == 0:
        print(
            f"loss: {full_loss.item():>7f} {reg.item():>7f}  [{epoch:>5d}/{n_epoch:>5d}]"
        )

plt.plot(all_losses)
plt.show()
