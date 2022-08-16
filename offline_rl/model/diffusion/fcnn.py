import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model import FCNN
from offline_rl.model.diffusion import DiffusionNet


class DiffusionFCNN(DiffusionNet):
    def __init__(self):
        super().__init__()

        self.model = FCNN([8, 32, 32, 7], last_activation=False)
        self.sample_shape = (7, 1, 1)

    def forward(self, x: th.Tensor, t: float, alpha_bar: float):
        inp = x.reshape((x.shape[0], 7))
        t_embeded = t * np.ones((x.shape[0], 1))
        t_embeded = th.Tensor(t_embeded)
        inp = th.concat([inp, t_embeded], axis=1)

        x0 = self.model(inp).reshape(x.shape)

        return x - np.sqrt(alpha_bar) * x0
