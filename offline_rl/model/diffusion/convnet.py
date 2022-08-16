import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model.diffusion import DiffusionNet


class DiffusionConvNet(DiffusionNet):
    def __init__(self):
        super().__init__()

        # self.resnet = SimpleConvNet()
        self.resnet = FullConvNet()

        self.sample_shape = (7, 8, 8)
        self.output_shape = (1,) + self.sample_shape
        self.feature_shape = (1, 1) + self.sample_shape[1:]

        xs = np.linspace(0, 1, self.sample_shape[-1])
        xs, ys = np.meshgrid(xs, xs)
        xs = xs.reshape(self.feature_shape)
        ys = ys.reshape(self.feature_shape)
        xs = self.embed(xs, 4, 1)
        ys = self.embed(ys, 4, 1)
        self.pos_th = th.Tensor(np.concatenate([xs, ys], axis=1))

    def calc_x0(self, x: th.Tensor, t: float, alpha_bar: float):
        t_embeded = self.embed(t * np.ones(self.feature_shape), 4, 1)
        t_embeded = th.Tensor(t_embeded)

        batch_ones = th.ones(x.shape[:1] + (1,) * len(self.sample_shape))

        inp = th.concat([x, self.pos_th * batch_ones, t_embeded * batch_ones], axis=1)

        return self.resnet(inp)

    def forward(self, x: th.Tensor, t: float, alpha_bar: float):
        return x - np.sqrt(alpha_bar) * self.calc_x0(x, t, alpha_bar)


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        sample_channels = 7
        in_channels = sample_channels + 8 + 16  # + 16  # 8 + 16
        compute_channels = 64
        kernel_size = 3

        self.layer_0 = nn.Conv2d(
            in_channels,
            compute_channels,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        self.layer_1 = nn.Conv2d(
            in_channels + compute_channels,
            compute_channels,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        self.layer_2 = nn.Conv2d(
            in_channels + compute_channels,
            sample_channels,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x):
        inp = x
        x = th.relu(self.layer_0(x))
        x = th.concat([x, inp], axis=1)
        x = th.relu(self.layer_1(x))
        x = th.concat([x, inp], axis=1)
        x = self.layer_2(x)

        return x


class FullConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        sample_channels = 7
        in_channels = sample_channels + 8 + 16  # + 16  # 8 + 16
        compute_channels = 64
        kernel_size = 3

        self.layer_0 = nn.Conv2d(
            in_channels,
            compute_channels,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        self.compute_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels + compute_channels,
                    compute_channels,
                    kernel_size,
                    padding="same",
                    padding_mode="reflect",
                )
                for i in range(10)
            ]
        )
        self.out_layer = nn.Conv2d(
            in_channels + compute_channels,
            sample_channels,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x):
        inp = x
        x = th.relu(self.layer_0(x))

        for layer in self.compute_layers:
            x = th.concat([x, inp], axis=1)
            x = th.relu(layer(x))

        x = th.concat([x, inp], axis=1)
        x = self.out_layer(x)

        return x
