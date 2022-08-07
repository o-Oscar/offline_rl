import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model.base import BaseModel


class FCNN(BaseModel):
    def __init__(self, sizes: list[int], last_activation: bool = True):
        super().__init__()

        self.sizes = sizes
        self.last_activation = last_activation

        self.linears = nn.ModuleList(
            [nn.Linear(inp, out) for inp, out in zip(self.sizes[:-1], self.sizes[1:])]
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, l in enumerate(self.linears):
            if (not self.last_activation) and i == len(self.sizes) - 2:
                x = l(x)
            else:
                x = self.activation(l(x))
        return x
