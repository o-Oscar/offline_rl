import numpy as np
from offline_rl.model.mdn.simple_model import SimpleMDN
from offline_rl.utils.dataset.synthetic import cross, curve, linear, step

# create model
model = SimpleMDN(1, 1)

# create dataset

# for some epoch
n_epoch = 10
for epoch in range(n_epoch):
    # train model

    # save model

    # report statistics
    print("coucou")
