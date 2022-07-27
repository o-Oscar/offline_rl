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

# create model
model = SimpleMDN(1, 1)
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)


# create dataset
xs, ys = torch_wrapper(one_dim_wrapper(linear))(1000)

# for some epoch
n_epoch = 1000
for epoch in range(n_epoch):
    # train model
    loss = th.sum(-model.log_density(xs, ys))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save model

    # report statistics
    if epoch % 1 == 0:
        loss = loss.item()
        print(f"loss: {loss:>7f}  [{epoch:>5d}/{n_epoch:>5d}]")
