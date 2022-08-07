from turtle import forward

import torch as th
import torch.nn as nn


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel, padding="same", padding_mode="reflect"
        ),
        nn.ReLU(inplace=True),
    )


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, compute_channels):
        super().__init__()

        self.layer_1 = convrelu(in_channels, compute_channels, 3, "symetric")
        self.layer_2 = convrelu(compute_channels, compute_channels, 3, "symetric")
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, inp):
        x = self.layer_1(inp)
        x = self.layer_2(x)
        x = th.concat([x, inp], dim=1)
        return x


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layer_1 = convrelu(in_channels, in_channels, 3, "symetric")
        self.layer_2 = convrelu(in_channels, in_channels, 3, "symetric")
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x, self.max_pool(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, in_channels):
        super().__init__()

        self.upsample = nn.Upsample(in_size * 2, mode="bilinear", align_corners=True)
        self.up_conv = nn.Conv2d(
            in_channels * 2, in_channels, 2, padding="same", padding_mode="reflect"
        )

        self.layer_1 = convrelu(in_channels * 2, in_channels, 3, "symetric")
        self.layer_2 = convrelu(in_channels, in_channels, 3, "symetric")

    def forward(self, x, y):
        y = self.upsample(y)
        y = self.up_conv(y)

        x = th.concat([x, y], dim=-3)
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.resnet = ResNetBlock(in_channels, in_channels)
        self.down = UNetDownBlock(in_channels * 2)

    def forward(self, x):
        # print(x.shape)
        # print(self.resnet(x).shape)
        # print(self.down(self.resnet(x))[0].shape)
        # print()
        return self.down(self.resnet(x))


class ResNetUpBlock(nn.Module):
    def __init__(self, in_size, in_channels):
        super().__init__()

        self.up = UNetUpBlock(in_size, in_channels)
        self.resnet = ResNetBlock(in_channels, in_channels)

    def forward(self, x, y):
        # print(x.shape, y.shape)
        # print(self.up(x, y).shape)
        # print(self.resnet(self.up(x, y)).shape)
        # print()
        return self.resnet(self.up(x, y))


class UNet(nn.Module):
    def __init__(self, in_size, in_channels, compute_channels, depth):
        super().__init__()

        assert in_size % (2**depth) == 0

        all_channel_nb = [compute_channels * (2**i) for i in range(depth)]
        all_sizes = [in_size // (2**i) for i in range(depth)]

        self.first_layer = convrelu(in_channels, compute_channels, 3, "symetric")

        self.down_layers = nn.ModuleList(
            [UNetDownBlock(in_ch) for in_ch in all_channel_nb]
        )

        self.up_layers = nn.ModuleList(
            [
                UNetUpBlock(in_si, in_ch)
                for in_si, in_ch in reversed(
                    list(zip(all_sizes[1:], all_channel_nb[1:]))
                )
            ]
        )

    def forward(self, x):
        x = self.first_layer(x)

        all_residuals = []
        for down_layer in self.down_layers:
            res, x = down_layer(x)
            all_residuals.append(res)

        x = all_residuals.pop()

        for res, up_layer in zip(reversed(all_residuals), self.up_layers):
            x = up_layer(res, x)

        return x


class ResUNet(nn.Module):
    def __init__(self, in_size, in_channels, compute_channels, depth):
        super().__init__()

        assert in_size % (2**depth) == 0

        all_channel_nb = [compute_channels * (2**i) for i in range(depth)]
        all_sizes = [in_size // (2**i) for i in range(depth)]

        self.first_layer = convrelu(in_channels, compute_channels, 3, "symetric")

        self.down_layers = nn.ModuleList(
            [ResNetDownBlock(in_ch) for in_ch in all_channel_nb]
        )

        self.up_layers = nn.ModuleList(
            [
                ResNetUpBlock(in_si, in_ch)
                for in_si, in_ch in reversed(
                    list(zip(all_sizes[1:], all_channel_nb[1:]))
                )
            ]
        )

    def forward(self, x):
        x = self.first_layer(x)

        all_residuals = []
        for down_layer in self.down_layers:
            res, x = down_layer(x)
            all_residuals.append(res)

        x = all_residuals.pop()

        for res, up_layer in zip(reversed(all_residuals), self.up_layers):
            x = up_layer(res, x)

        return x


if __name__ == "__main__":
    import numpy as np

    if True:
        model = UNet(16, 4, 2)
        inp = np.zeros((3, 4, 16, 16))
        inp = th.Tensor(inp)

        print("inp shape :", inp.shape)
        out = model(inp)
        print("out shape :", out.shape)

    if False:
        down_block = UNetDownBlock(16)
        down_block_end = UNetDownBlock(32)
        up_block = UNetUpBlock(8, 32)

        inp = np.zeros((3, 16, 8, 8))
        inp = th.Tensor(inp)

        print("input :", inp.shape)

        down, down_pool = down_block(inp)

        print("after first down loayer :", down.shape, down_pool.shape)

        down_end, _ = down_block_end(down_pool)

        print("after last down layer :", down_end.shape)

        up = up_block(down, down_end)

        print("after the up layer :", up.shape)
