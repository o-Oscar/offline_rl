import numpy as np
import torch as th
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()

        self.sizes = sizes
        self.linears = nn.ModuleList(
            [nn.Linear(inp, out) for inp, out in zip(self.sizes[:-1], self.sizes[1:])]
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.activation(l(x))
        return x
