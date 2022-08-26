import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model import graph_model as gm
from offline_rl.model.diffusion import DiffusionNet


class DiffusionAttentionNet(DiffusionNet):
    def __init__(self):
        super().__init__()

        batch = gm.Dimention("batch", -1)
        img_features = gm.Dimention("img_features", 7)
        img_width = gm.Dimention("img_width", 8)
        img_height = gm.Dimention("img_height", 8)
        pos_embedding = gm.Dimention("pos_embedding", 16 * 2)
        t_embedding = gm.Dimention("t_embedding", 16 * 2)
        computation_dim = gm.Dimention("computation_dim", 16)

        self.batch = batch
        self.t_embedding = t_embedding

        noisy = gm.Input("noisy", [batch, img_features, img_width, img_height])
        input = gm.Input("input", [batch, img_features, img_width, img_height])
        t_input = gm.Input("t", [t_embedding])
        xs_input = gm.Input("pos_x", [pos_embedding, img_width, img_height])
        ys_input = gm.Input("pos_y", [pos_embedding, img_width, img_height])

        t = gm.BlowUp(t_input, [batch, t_embedding, img_width, img_height])
        xs = gm.BlowUp(xs_input, [batch, pos_embedding, img_width, img_height])
        ys = gm.BlowUp(ys_input, [batch, pos_embedding, img_width, img_height])

        full_input = gm.Concatenate([noisy, input, t, xs, ys], 1)
        projected_input = gm.Linear(full_input, 1, computation_dim)
        layers = 1

        # # first block
        # attention = gm.SelfAttention(projected_input, [0], 1)
        # feed_forward = gm.ReLU(gm.Linear(attention, 1, computation_dim))

        # # remaining blocks
        # for i in range(layers - 1):
        #     block_input = gm.Concatenate([full_input, feed_forward], 1)
        #     projected_block = gm.Linear(block_input, 1, computation_dim)
        #     attention = gm.SelfAttention(projected_block, [0], 1)
        #     feed_forward = gm.ReLU(gm.Linear(attention, 1, computation_dim))

        # self.output = gm.Linear(feed_forward, 1, img_features)

        self.output = gm.Linear(input, 1, img_features)

        xs = np.linspace(0, 1, img_width.value)
        xs, ys = np.meshgrid(xs, xs)
        self.xs = th.Tensor(self.embed(xs, pos_embedding.value, 0))
        self.ys = th.Tensor(self.embed(ys, pos_embedding.value, 0))

        self.sample_shape = (img_features.value, img_width.value, img_height.value)

    def calc_x0(self, noisy: th.Tensor, t: float, alpha_bar: float, input: th.Tensor):
        self.batch.value = noisy.shape[0]

        t_embeded = self.embed(t, self.t_embedding.value, 0)
        t_embeded = th.Tensor(t_embeded)

        return (
            self.output(
                **{
                    "noisy": noisy,
                    "input": input,
                    "pos_x": self.xs,
                    "pos_y": self.ys,
                    "t": t,
                }
            )
            + input * 0
        )

    def forward(
        self, x: th.Tensor, t: float, alpha_bar: float, infos: th.Tensor = None
    ):
        return x - np.sqrt(alpha_bar) * self.calc_x0(x, t, alpha_bar, infos)
