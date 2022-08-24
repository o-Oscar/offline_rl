from turtle import forward

import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model.diffusion import DiffusionNet


class DiffusionAttentionNet(DiffusionNet):
    def __init__(self, nb_infos_channels=0):
        super().__init__()

        self.sample_shape = (7, 8, 8)
        self.output_shape = (1,) + self.sample_shape
        self.feature_shape = (1, 1) + self.sample_shape[1:]
        self.nb_infos_channels = nb_infos_channels

        xs = np.linspace(0, 1, self.sample_shape[-1])
        xs, ys = np.meshgrid(xs, xs)
        xs = xs.reshape(self.feature_shape)
        ys = ys.reshape(self.feature_shape)
        xs = self.embed(xs, 4, 1)
        ys = self.embed(ys, 4, 1)
        self.pos_th = th.Tensor(np.concatenate([xs, ys], axis=1))

        self.t_embed_channels = 4

        self.input_channels = (
            self.sample_shape[0]
            + self.pos_th.shape[1]
            + self.t_embed_channels * 2
            + nb_infos_channels
        )
        self.input_shape = (self.input_channels,) + self.sample_shape[1:]

        self.resnet = AttentionNet(self.input_shape, self.sample_shape[0])

    def calc_x0(
        self, x: th.Tensor, t: float, alpha_bar: float, infos: th.Tensor = None
    ):
        t_embeded = self.embed(
            t * np.ones(self.feature_shape), self.t_embed_channels, 1
        )
        t_embeded = th.Tensor(t_embeded)

        batch_ones = th.ones(x.shape[:1] + (1,) * len(self.sample_shape))

        assert (self.nb_infos_channels == 0) == (infos is None)
        if self.nb_infos_channels == 0:
            inp = th.concat(
                [x, self.pos_th * batch_ones, t_embeded * batch_ones], axis=1
            )
        else:
            inp = th.concat(
                [x, self.pos_th * batch_ones, t_embeded * batch_ones, infos], axis=1
            )
        return self.resnet(inp)

    def forward(
        self, x: th.Tensor, t: float, alpha_bar: float, infos: th.Tensor = None
    ):
        return x - np.sqrt(alpha_bar) * self.calc_x0(x, t, alpha_bar, infos)


# def get_feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
#     return nn.Sequential(
#         nn.Linear(dim_input, dim_feedforward),
#         nn.ReLU(),
#         nn.Linear(dim_feedforward, dim_input),
#     )
def get_feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            dim_input,
            dim_feedforward,
            3,
            padding="same",
            padding_mode="reflect",
        ),
        nn.ReLU(),
        nn.Conv2d(
            dim_feedforward,
            dim_input,
            3,
            padding="same",
            padding_mode="reflect",
        ),
    )


class CustomMultiheadAttention(nn.MultiheadAttention):
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
    ):
        to_return = super().forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        return to_return[0]


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.0):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp: th.Tensor, tensors: list[th.Tensor]) -> th.Tensor:
        return self.norm(inp + self.dropout(self.sublayer(*tensors)))


class TransformerBlock(nn.Module):
    def __init__(self, compute_dims, to_transformer_view, from_transformer_view):
        super().__init__()

        self.compute_dims = compute_dims

        num_heads = 4
        attention = CustomMultiheadAttention(
            self.compute_dims, num_heads, batch_first=True
        )
        self.attention = Residual(attention, self.compute_dims)

        feed_forward = get_feed_forward(self.compute_dims, self.compute_dims * 4)
        self.feed_forward = Residual(feed_forward, (self.compute_dims, 8, 8))

        self.to_transformer_view = to_transformer_view
        self.from_transformer_view = from_transformer_view

    def forward(self, x: th.Tensor, skip: th.Tensor):

        x = x.view(self.to_transformer_view)
        x = x.transpose(1, 2)

        x = self.attention(x, [x, x, x])

        x = x.transpose(1, 2)
        x = x.view(self.from_transformer_view)

        x = self.feed_forward(x, [x])

        return x


class AttentionNet(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()

        self.input_shape = input_shape
        self.out_channels = out_channels
        self.output_shape = (out_channels,) + input_shape[1:]

        in_channels = self.input_shape[0]
        self.compute_channels = 16
        # kernel_size = 3

        self.compute_shape = (self.compute_channels, input_shape[1] * input_shape[2])
        self.to_transformer_view = (-1,) + self.compute_shape
        self.from_transformer_view = (-1, self.compute_channels) + self.input_shape[1:]

        self.layer_0 = nn.Conv2d(
            in_channels,
            self.compute_channels,
            3,
            padding="same",
            padding_mode="reflect",
        )

        self.attention_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.compute_channels,
                    self.to_transformer_view,
                    self.from_transformer_view,
                )
                for i in range(3)
            ]
        )

        self.layer_2 = nn.Conv2d(
            self.compute_channels,
            self.out_channels,
            3,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x: th.Tensor):
        # inp = x
        inp = self.layer_0(x)

        # x = x.view((-1,) + self.compute_shape)
        # x = x.transpose(1, 2)
        for attention_block in self.attention_blocks:
            x = attention_block(x, inp)
        # x = x.transpose(1, 2)
        # x = x.view((-1, self.compute_channels) + self.input_shape[1:])

        x = self.layer_2(x)

        return x
