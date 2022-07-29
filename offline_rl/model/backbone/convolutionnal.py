import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CNN2d(nn.Module):
    def __init__(self, channel_sizes: list[int], kernel_sizes: list[int]):
        super().__init__()

        self.channel_sizes = channel_sizes
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_size, out_size, kernel_size)
                for in_size, out_size, kernel_size in zip(
                    channel_sizes[:-1], channel_sizes[1:], kernel_sizes
                )
            ]
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.pool(self.activation(l(x)))
        return x
