from turtle import forward

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
                nn.Conv2d(in_size, out_size, kernel_size, padding="same")
                for in_size, out_size, kernel_size in zip(
                    channel_sizes[:-1], channel_sizes[1:], kernel_sizes
                )
            ]
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor):
        # x = x.transpose(-3, -1)
        for i, l in enumerate(self.convs):
            x = self.pool(self.activation(l(x)))
        # x = x.transpose(-3, -1)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, res_channels, compute_channels, out_channels):
        super().__init__()

        self.res_channels = res_channels
        self.layer_1 = nn.Conv2d(
            in_channels + res_channels,
            compute_channels,
            3,
            padding="same",
            padding_mode="reflect",
        )
        self.layer_2 = nn.Conv2d(
            compute_channels, out_channels, 3, padding="same", padding_mode="reflect"
        )
        self.activation = nn.ReLU()

    def forward(self, x: th.Tensor, residual: list[th.Tensor] = None):
        assert (residual is None) == (self.res_channels == 0)

        if self.res_channels == 0:
            features = x
        else:
            features = th.concat([x] + residual, axis=1)
        features = self.activation(self.layer_1(features))
        return self.activation(self.layer_2(features))


# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         in_channels = 8
#         compute_channels = 32
#         out_channels = in_channels - 1

#         self.initial_layer = nn.Conv2d(
#             in_channels, compute_channels, 1, padding="same", padding_mode="reflect"
#         )

#         blocks = []
#         for i in range(4):
#             if i == 0:
#                 layer_channels = compute_channels
#                 res_channels = 0
#             else:
#                 layer_channels = compute_channels
#                 res_channels = compute_channels

#             layer = ResNetBlock(
#                 layer_channels, res_channels, compute_channels, compute_channels
#             )
#             blocks.append(layer)

#         self.blocks = nn.ModuleList(blocks)

#         self.final_layer = nn.Conv2d(
#             compute_channels, out_channels, 1, padding="same", padding_mode="reflect"
#         )

#     def forward(self, x):

#         x = th.relu(self.initial_layer(x))
#         res = x
#         x = self.blocks[0](x)

#         for i, block in enumerate(self.blocks[1:]):
#             prev_x = x
#             x = block(x, [res])
#             res = prev_x

#         return self.final_layer(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 8
        compute_channels = 32
        out_channels = in_channels - 1

        self.initial_layer = nn.Conv2d(
            in_channels, compute_channels, 1, padding="same", padding_mode="reflect"
        )

        blocks = []
        blocks.append(
            ResNetBlock(compute_channels, 0, compute_channels, compute_channels)
        )
        for i in range(3):
            blocks.append(
                ResNetBlock(
                    compute_channels,
                    compute_channels,
                    compute_channels,
                    compute_channels,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = nn.Conv2d(
            compute_channels, out_channels, 1, padding="same", padding_mode="reflect"
        )

    def forward(self, x):

        x = th.relu(self.initial_layer(x))

        x0 = self.blocks[0](x)
        for i in range(3):
            x0 = self.blocks[i + 1](x0, [x])

        return self.final_layer(x0)


# class SimpleResNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         in_channels = (
#             7 + 1
#         )  # TODO : add t as a bunch of sinusoids (transformer embedding)
#         compute_channels = 64
#         out_channels = in_channels - 1

#         self.layer_0 = nn.Conv2d(
#             in_channels, compute_channels, 3, padding="same", padding_mode="reflect"
#         )

#         self.layer_1 = nn.Conv2d(
#             compute_channels,
#             compute_channels,
#             3,
#             padding="same",
#             padding_mode="reflect",
#         )
#         self.layer_2 = nn.Conv2d(
#             compute_channels,
#             out_channels,
#             3,
#             padding="same",
#             padding_mode="reflect",
#         )

#     def forward(self, x):

#         x = th.relu(self.layer_0(x))
#         x = th.relu(self.layer_1(x))
#         x = self.layer_2(x)

#         return x

import numpy as np


class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 7
        t_channels = 8
        pos_channels = 16  # 2
        compute_channels = 64
        out_channels = in_channels
        layer_in_channels = t_channels + in_channels + pos_channels

        self.layer_0 = nn.Conv2d(
            layer_in_channels,
            compute_channels,
            3,
            padding="same",
            padding_mode="reflect",
        )

        self.layer_1 = nn.Conv2d(
            layer_in_channels + compute_channels,
            compute_channels,
            3,
            padding="same",
            padding_mode="reflect",
        )
        self.layer_2 = nn.Conv2d(
            layer_in_channels + compute_channels,
            out_channels,
            3,
            padding="same",
            padding_mode="reflect",
        )

        xs = np.linspace(0, 1, 8)
        xs, ys = np.meshgrid(xs, xs)
        xs = xs.reshape((1, 1, 8, 8))
        ys = ys.reshape((1, 1, 8, 8))
        pos = np.concatenate([xs, ys], axis=1)
        all_pos = []
        for i in range(4):
            phase = np.pi * xs * (2**i)
            all_pos.append(np.sin(phase))
            all_pos.append(np.cos(phase))
        for i in range(4):
            phase = np.pi * ys * (2**i)
            all_pos.append(np.sin(phase))
            all_pos.append(np.cos(phase))
        pos = np.concatenate(all_pos, axis=1)
        self.pos_th = th.Tensor(pos)

    def forward(self, x, t):
        ones = np.ones((x.shape[0], 1, 1, 1), dtype=np.float32)
        pos = self.pos_th * ones
        inp = x
        x = th.concat([inp, t, pos], axis=1)
        x = th.relu(self.layer_0(x))
        x = th.concat([x, inp, t, pos], axis=1)
        x = th.relu(self.layer_1(x))
        x = th.concat([x, inp, t, pos], axis=1)
        x = self.layer_2(x)

        return x
